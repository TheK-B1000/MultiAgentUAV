from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from rl.global_state import GLOBAL_STATE_DIM, GLOBAL_STATE_V6I7_DIM
from rl.latent_marl import (
    CONTEXT_STATE_DIM,
    LatentConditionedActor,
    RecurrentSelectorCell,
    StrategyEncoder,
)
from rl.latent_phase_labels import TEAM_PHASES
from rl.networks import CNNEncoder, CentralizedCritic


def _validate_indices(values: torch.Tensor, upper_bound: int, name: str) -> torch.Tensor:
    """Validate categorical indices and raise on out-of-range values."""
    if upper_bound <= 0:
        raise ValueError(f"{name}: upper_bound must be positive, got {upper_bound}")
    flat = values.long().reshape(-1)
    if torch.any(flat < 0) or torch.any(flat >= upper_bound):
        lo = int(flat.min().item())
        hi = int(flat.max().item())
        raise ValueError(
            f"{name} must be in [0, {upper_bound - 1}], got range [{lo}, {hi}]"
        )
    return flat


def _migrate_action_conditioned_critic_weights(
    state_dict: Dict[str, Any],
    *,
    prefix: str,
    global_state_dim: int,
    joint_action_dim: int,
    latent_k: int,
) -> Dict[str, Any]:
    """Drop joint-action columns from legacy action-conditioned critic checkpoints."""
    if latent_k <= 0 or joint_action_dim <= 0:
        return state_dict
    key = prefix + "critic.net.0.weight"
    if key not in state_dict:
        return state_dict
    weight = state_dict[key]
    old_in = int(global_state_dim + joint_action_dim + latent_k)
    new_in = int(global_state_dim + latent_k)
    if int(weight.shape[1]) != old_in:
        return state_dict
    migrated = torch.cat(
        [weight[:, :global_state_dim], weight[:, global_state_dim + joint_action_dim :]],
        dim=1,
    )
    out = dict(state_dict)
    out[key] = migrated
    bias_key = prefix + "critic.net.0.bias"
    if bias_key in out:
        out[bias_key] = state_dict[bias_key]
    return out


# Legacy state-dict keys mapped to the new composed submodule paths under
# ``latent_actor``. Pre-composition checkpoints stored these tensors directly
# at the top of ``SharedActorCentralizedCritic``; the model now owns them via
# ``self.latent_actor = LatentConditionedActor(...)``. The remap keeps every
# checkpoint on disk loadable without manual migration.
_LEGACY_ACTOR_RENAMES: tuple[tuple[str, str], ...] = (
    ("actor_body.", "latent_actor.body."),
    ("actor_head.", "latent_actor.action_head."),
    ("strategy_embedding.", "latent_actor.strategy_embedding."),
)


def remap_legacy_actor_state_dict_keys(
    state_dict: Dict[str, Any], *, prefix: str = ""
) -> Dict[str, Any]:
    """Rewrite pre-composition actor/embedding keys to their composed paths.

    ``prefix`` is the path to the :class:`SharedActorCentralizedCritic` inside
    the checkpoint (empty string when the model is at the top of the file, or
    e.g. ``"model."`` when nested). Keys that already use the new ``latent_actor.*``
    layout pass through untouched, so this remap is idempotent.
    """
    if not state_dict:
        return dict(state_dict)
    out: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if prefix and not key.startswith(prefix):
            out[key] = value
            continue
        tail = key[len(prefix):] if prefix else key
        new_tail = tail
        for old, new in _LEGACY_ACTOR_RENAMES:
            if tail.startswith(old):
                new_tail = new + tail[len(old):]
                break
        out[prefix + new_tail] = value
    return out


def _migrate_legacy_aliased_strategy_modules(
    state_dict: Dict[str, Any],
    *,
    prefix: str = "",
    has_strategy_encoder: bool,
    has_strategy_aux_return_head: bool,
) -> Dict[str, Any]:
    """Bridge the Step-5 split between ``StrategyEncoder`` and the aux-return head.

    Pre-Step-5 ``SharedActorCentralizedCritic`` aliased a single
    :class:`StrategyEncoder` instance to **either** ``self.strategy_encoder``
    (aux-return head off) **or** ``self.strategy_aux_return_head`` (aux-return
    head on), so the same module silently served two distinct roles. After
    Step 5 each role has its own ``nn.Module``; this helper rewrites legacy
    checkpoints accordingly:

    * Aux head was **off** on disk → state dict already has ``strategy_encoder.*``
      and no aux-head keys; pass through.
    * Aux head was **on** on disk and the new model still keeps it on → state
      dict has ``strategy_aux_return_head.*`` only; mirror those weights into
      ``strategy_encoder.*`` (so q_phi(z|s) keeps the trained behavior the
      aliased module was producing) and keep the aux-head weights too.
    * Aux head was on on disk but the new model has it off → rename the legacy
      aux-head weights into ``strategy_encoder.*`` (single role survives).

    The migration is idempotent: a state dict that already matches the new
    layout passes through untouched.
    """
    if not state_dict:
        return dict(state_dict)

    enc_full_prefix = prefix + "strategy_encoder."
    aux_full_prefix = prefix + "strategy_aux_return_head."

    legacy_aux_keys = [
        k for k in state_dict
        if k.startswith(aux_full_prefix) or k == aux_full_prefix[:-1]
    ]
    has_canonical_encoder = any(
        k.startswith(enc_full_prefix) or k == enc_full_prefix[:-1] for k in state_dict
    )

    out: Dict[str, Any] = dict(state_dict)
    if not legacy_aux_keys:
        return out
    if has_canonical_encoder:
        enc_first_key = enc_full_prefix + "net.0.weight"
        aux_first_key = aux_full_prefix + "net.0.weight"
        enc_weight = state_dict.get(enc_first_key)
        aux_weight = state_dict.get(aux_first_key)
        if not has_strategy_aux_return_head:
            for k in legacy_aux_keys:
                out.pop(k, None)
        elif (
            enc_weight is not None
            and aux_weight is not None
            and tuple(enc_weight.shape) != tuple(aux_weight.shape)
        ):
            for k in legacy_aux_keys:
                out.pop(k, None)
        return out

    enc_first_key = enc_full_prefix + "net.0.weight"
    aux_first_key = aux_full_prefix + "net.0.weight"
    enc_weight = state_dict.get(enc_first_key)
    aux_weight = state_dict.get(aux_first_key)
    if (
        enc_weight is not None
        and aux_weight is not None
        and tuple(enc_weight.shape) != tuple(aux_weight.shape)
    ):
        # Recurrent selector checkpoints need a wider encoder input than the
        # legacy aux-only head; keep canonical encoder keys and drop aux aliases.
        for k in legacy_aux_keys:
            out.pop(k, None)
        return out

    # Mirror legacy aux-head weights into strategy_encoder so q_phi(z|s) is
    # initialized from what the aliased module had been producing.
    for k in legacy_aux_keys:
        if k.startswith(aux_full_prefix):
            mirror_key = enc_full_prefix + k[len(aux_full_prefix):]
        else:
            mirror_key = enc_full_prefix[:-1]
        out[mirror_key] = state_dict[k]

    if not has_strategy_aux_return_head:
        # Model dropped the aux head; the mirrored copy already holds the
        # legacy weights as strategy_encoder, so the originals are surplus.
        for k in legacy_aux_keys:
            del out[k]
    # ``has_strategy_encoder=False`` cannot reach this branch since latent
    # strategy is either off (no migration needed) or on (encoder is always
    # present after Step 5).
    return out


class SharedActorCentralizedCritic(nn.Module):
    """Shared decentralized actor with an optional latent team strategy."""

    def __init__(
        self,
        observation_space,
        action_space,
        *,
        actor_hidden_dim: int = 256,
        actor_cnn_feature_dim: int = 128,
        critic_hidden_dim: int = 128,
        latent_k: int = 0,
        z_embed_dim: int = 16,
        strategy_hidden_dim: int = 128,
        use_strategy_aux_return_head: bool = False,
        use_episode_strategy_value_head: bool = False,
        use_recurrent_selector: bool = False,
        recurrent_selector_hidden_dim: int = 32,
        strategy_tau: float = 1.0,
        latent_actor_z_onehot_enabled: bool = False,
        latent_actor_z_onehot_scale: float = 1.0,
        latent_actor_z_embed_scale: float = 1.0,
        latent_actor_z_adapter_enabled: bool = False,
        latent_actor_z_adapter_scale: float = 0.0,
        latent_actor_z_adapter_init_std: float = 0.02,
        latent_actor_z_film_layers: int = 1,
        enable_actor_z_film: bool = False,
        actor_z_film_init_scale: float = 0.0,
        actor_z_film_layer: int = 2,
        latent_actor_conditioning: str = "concat",
        enable_latent_z_residual: bool = False,
        latent_z_gate_init: float = 0.01,
        communication_enabled: bool = False,
        comm_num_symbols: int = 4,
        experiment_id: str = "",
        router_context_mode: str = "",
        router_context_dimension: int = 0,
    ) -> None:
        super().__init__()
        grid_shape = tuple(int(v) for v in observation_space.spaces["grid"].shape)
        vec_shape = tuple(int(v) for v in observation_space.spaces["vec"].shape)
        if len(grid_shape) != 4:
            raise ValueError(f"Expected tokenized grid shape (N, C, H, W), got {grid_shape!r}")
        if len(vec_shape) != 2:
            raise ValueError(f"Expected tokenized vec shape (N, V), got {vec_shape!r}")

        self.n_agents = int(grid_shape[0])
        self.vec_dim = int(vec_shape[1])
        c, h, w = int(grid_shape[1]), int(grid_shape[2]), int(grid_shape[3])
        self.grid_shape = (c, h, w)
        self.actor_cnn = CNNEncoder(self.grid_shape, feature_dim=int(actor_cnn_feature_dim))
        self.actor_cnn_feature_dim = int(self.actor_cnn.feature_dim)
        self._scalar_per_agent = self.vec_dim
        self._local_actor_in_dim = self.actor_cnn_feature_dim + self._scalar_per_agent
        self.action_dims = tuple(int(v) for v in getattr(action_space, "nvec", []))
        if len(self.action_dims) % self.n_agents != 0:
            raise ValueError("MultiDiscrete action heads must divide evenly across agents.")
        self.heads_per_agent = len(self.action_dims) // self.n_agents
        self.per_agent_action_dims = self.action_dims[: self.heads_per_agent]
        for idx in range(self.n_agents):
            start = idx * self.heads_per_agent
            end = start + self.heads_per_agent
            if self.action_dims[start:end] != self.per_agent_action_dims:
                raise ValueError("All agents must share the same macro/target action heads.")
        self.per_agent_logits = int(sum(self.per_agent_action_dims))
        self.joint_action_onehot_dim = int(sum(self.action_dims))
        self.latent_k = max(0, int(latent_k))
        self.uses_latent_strategy = self.latent_k > 0
        self.z_embed_dim = int(z_embed_dim) if self.uses_latent_strategy else 0
        self.z_onehot_dim = (
            int(self.latent_k)
            if self.uses_latent_strategy and bool(latent_actor_z_onehot_enabled)
            else 0
        )
        self.use_strategy_aux_return_head = bool(use_strategy_aux_return_head) and self.uses_latent_strategy
        self.use_episode_strategy_value_head = bool(use_episode_strategy_value_head) and self.uses_latent_strategy
        self.use_recurrent_selector = bool(use_recurrent_selector) and self.uses_latent_strategy
        self.recurrent_selector_hidden_dim = (
            max(1, int(recurrent_selector_hidden_dim)) if self.use_recurrent_selector else 0
        )
        self.strategy_tau = max(1e-3, float(strategy_tau))
        self.experiment_id = str(experiment_id or "")
        self.router_context_mode = str(router_context_mode or "")
        self.router_context_dimension = int(router_context_dimension or 0)
        self.router_current_plus_delta_enabled = self.router_context_mode == "current_plus_delta"

        # V6I7: router_context_mode="current" disables EMA stack and uses raw
        # global state (augmented with scheduler phase to 35 dims) as input.
        if self.router_context_mode == "current" and self.uses_latent_strategy:
            self.global_state_dim = GLOBAL_STATE_V6I7_DIM
        elif self.uses_latent_strategy:
            self.global_state_dim = CONTEXT_STATE_DIM
        else:
            self.global_state_dim = GLOBAL_STATE_DIM

        q_phi_input_dim = (
            self.router_context_dimension
            if self.router_current_plus_delta_enabled and self.uses_latent_strategy
            else int(self.global_state_dim)
        )
        if self.use_recurrent_selector and not self.router_current_plus_delta_enabled:
            q_phi_input_dim += int(self.recurrent_selector_hidden_dim)
        self.q_phi_input_dim = q_phi_input_dim

        if self.uses_latent_strategy:
            if self.use_recurrent_selector:
                # GRU always takes raw 34-dim global state — not the augmented
                # 35-dim V6I7 state, since the scheduler phase is for the critic
                # (Markov property) and the GRU captures temporal info via h_t.
                self.selector_gru = RecurrentSelectorCell(
                    input_dim=GLOBAL_STATE_DIM,
                    hidden_dim=int(self.recurrent_selector_hidden_dim),
                )
            else:
                self.selector_gru = None
            # Step 5: ``StrategyEncoder`` (q_phi(z|s), the latent team-strategy policy)
            # and the optional A2 auxiliary per-z return regression head are now
            # ALWAYS distinct ``nn.Module`` instances when both are enabled. Before
            # this change a single ``StrategyEncoder`` was aliased to either slot
            # depending on the cfg flag, which made the same code path mean
            # different things at runtime ("q_phi is the aux head, sometimes").
            # See ``_migrate_legacy_aliased_strategy_modules`` for the on-disk
            # checkpoint migration.
            self.strategy_encoder = StrategyEncoder(
                state_dim=self.q_phi_input_dim,
                latent_k=self.latent_k,
                hidden=int(strategy_hidden_dim),
            )
            if self.use_strategy_aux_return_head:
                self.strategy_aux_return_head = StrategyEncoder(
                    state_dim=self.global_state_dim,
                    latent_k=self.latent_k,
                    hidden=int(strategy_hidden_dim),
                )
            else:
                self.strategy_aux_return_head = None
            self.phase_predictor = nn.Linear(self.z_embed_dim, len(TEAM_PHASES))
        else:
            self.strategy_encoder = None
            self.selector_gru = None
            self.strategy_aux_return_head = None
            self.phase_predictor = None

        # Decentralized policy: CNN(grid) is concatenated with per-agent scalar features (+ z_emb), never `GLOBAL_STATE_DIM`.
        self._decentralized_actor_in_dim = int(
            self._local_actor_in_dim
            + (self.z_embed_dim if self.uses_latent_strategy else 0)
            + self.z_onehot_dim
        )
        # The decentralized actor body, output head, and strategy embedding are
        # owned by the canonical ``LatentConditionedActor``. Code that reads
        # ``self.actor_body`` / ``self.actor_head`` / ``self.strategy_embedding``
        # goes through the property shims below; legacy on-disk state dicts are
        # migrated by ``remap_legacy_actor_state_dict_keys``.
        self.latent_actor = LatentConditionedActor(
            local_feature_dim=int(self._local_actor_in_dim),
            latent_k=self.latent_k if self.uses_latent_strategy else 0,
            action_dim=int(self.per_agent_logits),
            z_embed_dim=self.z_embed_dim if self.uses_latent_strategy else 0,
            hidden_dim=int(actor_hidden_dim),
            z_onehot_enabled=bool(latent_actor_z_onehot_enabled),
            z_onehot_scale=float(latent_actor_z_onehot_scale),
            z_embed_scale=float(latent_actor_z_embed_scale),
            z_adapter_enabled=bool(latent_actor_z_adapter_enabled),
            z_adapter_scale=float(latent_actor_z_adapter_scale),
            z_adapter_init_std=float(latent_actor_z_adapter_init_std),
            z_film_layers=int(latent_actor_z_film_layers),
            enable_actor_z_film=bool(enable_actor_z_film),
            actor_z_film_init_scale=float(actor_z_film_init_scale),
            actor_z_film_layer=int(actor_z_film_layer),
            latent_actor_conditioning=latent_actor_conditioning,
            enable_latent_z_residual=bool(enable_latent_z_residual),
            latent_z_gate_init=float(latent_z_gate_init),
        )
        critic_extra_dim = self.latent_k if self.uses_latent_strategy else 0
        self.critic = CentralizedCritic(
            global_state_dim=self.global_state_dim,
            hidden_dim=int(critic_hidden_dim),
            extra_dim=critic_extra_dim,
        )
        if self.use_episode_strategy_value_head:
            episode_value_in = int(self.q_phi_input_dim + self.latent_k)
            self.episode_strategy_value_head = nn.Sequential(
                nn.Linear(episode_value_in, int(critic_hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(critic_hidden_dim), int(critic_hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(critic_hidden_dim), 1),
            )
        else:
            self.episode_strategy_value_head = None
        self.q_phi_input_dim = self._strategy_context_dim()
        self.critic_context_dim = int(self.critic.global_state_dim)
        self.critic_z_dim = int(self.latent_k) if self.uses_latent_strategy else 0
        self.critic_joint_action_dim = 0
        self.actor_input_dim = int(self._decentralized_actor_in_dim)
        self.communication_enabled = bool(communication_enabled)
        self.comm_num_symbols = max(1, int(comm_num_symbols)) if self.communication_enabled else 0
        self._message_head_in_dim = int(self._local_actor_in_dim)
        if self.uses_latent_strategy:
            self._message_head_in_dim += int(self.z_embed_dim)
        if self.communication_enabled:
            self.message_head = nn.Linear(self._message_head_in_dim, int(self.comm_num_symbols))
        else:
            self.message_head = None
        self._assert_input_contracts()
        # Optional: separate ``torch.Generator`` streams so q_\phi(z|s) sampling does not advance
        # the same RNG as per-head action Categoricals (fairer E3 vs no-latent; see docs).
        self._sampling_gen_strategy: Optional[torch.Generator] = None
        self._sampling_gen_action: Optional[torch.Generator] = None

    # ------------------------------------------------------------------
    # Legacy attribute shims. These point into ``self.latent_actor`` so any
    # test, diagnostic, or external caller still written against the old
    # attribute names continues to work. The shims read-only; do not assign.
    # ------------------------------------------------------------------

    @property
    def actor_body(self) -> nn.Module:
        """Composed actor MLP trunk (``latent_actor.body``); legacy alias."""
        return self.latent_actor.body

    @property
    def actor_head(self) -> nn.Module:
        """Composed actor output projection (``latent_actor.action_head``); legacy alias."""
        return self.latent_actor.action_head

    @property
    def strategy_embedding(self) -> Optional[nn.Embedding]:
        """Composed strategy embedding (``latent_actor.strategy_embedding``); ``None`` when no latent."""
        return self.latent_actor.strategy_embedding

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # Rewrite the slice of ``state_dict`` that targets this module so legacy
        # pre-composition keys (``actor_body.*``, ``actor_head.*``,
        # ``strategy_embedding.*``) load into the composed ``latent_actor``
        # submodule. The remap is idempotent: keys that already use the new
        # layout pass through untouched.
        has_legacy = any(
            key.startswith(prefix + old)
            for key in state_dict
            for old, _ in _LEGACY_ACTOR_RENAMES
        )
        if has_legacy:
            remapped = remap_legacy_actor_state_dict_keys(state_dict, prefix=prefix)
            state_dict.clear()
            state_dict.update(remapped)
        # Step 5: also bridge legacy aliased q_phi / aux-return head weights so
        # checkpoints saved when the aux-return head was the *same* module as
        # ``strategy_encoder`` still populate the new separate modules.
        migrated = _migrate_legacy_aliased_strategy_modules(
            state_dict,
            prefix=prefix,
            has_strategy_encoder=self.strategy_encoder is not None,
            has_strategy_aux_return_head=self.strategy_aux_return_head is not None,
        )
        migrated = _migrate_action_conditioned_critic_weights(
            migrated,
            prefix=prefix,
            global_state_dim=int(self.global_state_dim),
            joint_action_dim=int(self.joint_action_onehot_dim),
            latent_k=int(self.latent_k),
        )
        state_dict.clear()
        state_dict.update(migrated)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _strategy_context_dim(self) -> int:
        if not self.uses_latent_strategy:
            return 0
        # ``strategy_encoder`` (q_phi(z|s)) is always present when latent is on
        # since Step 5; the aux-return head, when enabled, is a separate module
        # with the same input contract.
        source = self.strategy_encoder
        if source is None:
            raise AssertionError("latent strategy enabled but q_phi module is missing")
        dim = getattr(source, "state_dim", None)
        if dim is not None:
            return int(dim)
        first = getattr(source, "net", [None])[0]
        if isinstance(first, nn.Linear):
            return int(first.in_features)
        raise AssertionError("could not resolve q_phi input dim")

    def _assert_input_contracts(self) -> None:
        actor_expected = int(self.actor_cnn_feature_dim) + int(self._scalar_per_agent)
        if self.uses_latent_strategy:
            actor_expected += int(self.z_embed_dim) + int(self.z_onehot_dim)
            # V6I7 "current" mode uses GLOBAL_STATE_V6I7_DIM (35); other latent
            # modes use CONTEXT_STATE_DIM (170).
            expected_global_dim = (
                int(GLOBAL_STATE_V6I7_DIM)
                if self.router_context_mode == "current"
                else int(CONTEXT_STATE_DIM)
            )
            if int(self.global_state_dim) != expected_global_dim:
                raise ValueError(
                    f"latent global_state_dim must be {expected_global_dim} "
                    f"(router_context_mode={self.router_context_mode!r}), "
                    f"got {self.global_state_dim}"
                )
            expected_q_phi_dim = (
                int(self.router_context_dimension)
                if self.router_current_plus_delta_enabled
                else int(expected_global_dim)
            )
            if self.use_recurrent_selector and not self.router_current_plus_delta_enabled:
                expected_q_phi_dim += int(self.recurrent_selector_hidden_dim)
            if int(self.q_phi_input_dim) != expected_q_phi_dim:
                raise ValueError(
                    f"q_phi_input_dim must be {expected_q_phi_dim}, got {self.q_phi_input_dim}"
                )
            if int(self.critic.global_state_dim) != expected_global_dim:
                raise ValueError(
                    f"critic global_state_dim must be {expected_global_dim}, got {self.critic.global_state_dim}"
                )
            if int(self._decentralized_actor_in_dim) != actor_expected:
                raise ValueError(
                    f"latent actor input dim must be {actor_expected}, got {self._decentralized_actor_in_dim}"
                )
            expected_extra = int(self.latent_k)
            if int(self.critic.extra_dim) != expected_extra:
                raise ValueError(
                    f"critic extra_dim must be latent_k = {expected_extra}, got {self.critic.extra_dim}"
                )
        else:
            if int(self.global_state_dim) != int(GLOBAL_STATE_DIM):
                raise ValueError(
                    f"no-latent global_state_dim must be {GLOBAL_STATE_DIM}, got {self.global_state_dim}"
                )
            if int(self.critic.global_state_dim) != int(GLOBAL_STATE_DIM):
                raise ValueError(
                    f"no-latent critic global_state_dim must be {GLOBAL_STATE_DIM}, got {self.critic.global_state_dim}"
                )
            if int(self.critic.extra_dim) != 0:
                raise ValueError(f"no-latent critic extra_dim must be 0, got {self.critic.extra_dim}")

        if int(self._decentralized_actor_in_dim) != actor_expected:
            raise ValueError(
                f"actor_input_dim={self._decentralized_actor_in_dim} must equal local obs + z embedding width "
                f"{actor_expected}"
            )
        first_actor = self.actor_body[0]
        if not isinstance(first_actor, nn.Linear) or int(first_actor.in_features) != actor_expected:
            got = getattr(first_actor, "in_features", None)
            raise ValueError(f"actor MLP first layer input {got} != decentralized actor input {actor_expected}")
        if int(self._decentralized_actor_in_dim) == int(CONTEXT_STATE_DIM):
            raise ValueError(
                f"actor_input_dim={self._decentralized_actor_in_dim} equals temporal_context_dim={CONTEXT_STATE_DIM}; "
                "actor must consume local obs + z embedding only, never the centralized temporal context."
            )

    def input_dim_contract(self) -> dict[str, int]:
        self._assert_input_contracts()
        rmode = str(self.router_context_mode or "")
        return {
            "base_global_state_dim": int(GLOBAL_STATE_DIM),
            "temporal_context_dim": int(CONTEXT_STATE_DIM),
            # For V6I7 (router_context_mode="current"): global_state_dim=35 (raw+phase),
            # recurrent_selector_hidden_dim=64, q_phi_input_dim=99 (35+64).
            # For EMA-stack modes: global_state_dim=170, q_phi_input_dim=170 (no GRU concat).
            "router_context_mode": rmode,
            "router_global_state_dim": int(self.global_state_dim),
            "recurrent_selector_hidden_dim": int(self.recurrent_selector_hidden_dim),
            "q_phi_input_dim": int(self.q_phi_input_dim),
            "critic_context_dim": int(self.critic_context_dim),
            "actor_input_dim": int(self.actor_input_dim),
            "actor_z_embed_dim": int(self.z_embed_dim),
            "actor_z_onehot_dim": int(self.z_onehot_dim),
            "actor_z_residual_enabled": int(
                bool(getattr(self.latent_actor, "enable_latent_z_residual", False))
            ),
            "critic_extra_dim": int(self.critic.extra_dim),
            "critic_z_dim": int(self.critic_z_dim),
            "critic_joint_action_dim": int(self.critic_joint_action_dim),
        }

    def log_architecture_summary(self) -> None:
        """Print one authoritative dimension decomposition to stdout."""
        c = self.input_dim_contract()
        rmode = c["router_context_mode"] or "ema_stack"
        actor_la = self.latent_actor
        print(f"[arch] router_context_mode={rmode!r}")
        print(f"[arch] base_global_state_dim={c['base_global_state_dim']}  "
              f"(GLOBAL_STATE_DIM, raw env features)")
        if rmode == "current":
            print(f"[arch] router_global_state_dim={c['router_global_state_dim']}  "
                  f"(raw {c['base_global_state_dim']} + 1 scheduler phase)")
            print(f"[arch] recurrent_selector_hidden_dim={c['recurrent_selector_hidden_dim']}")
            print(f"[arch] q_phi_input_dim={c['q_phi_input_dim']}  "
                  f"(={c['router_global_state_dim']}+{c['recurrent_selector_hidden_dim']})")
        else:
            print(f"[arch] temporal_context_dim={c['temporal_context_dim']}  "
                  f"(EMA stack: 5×{c['base_global_state_dim']})")
            print(f"[arch] q_phi_input_dim={c['q_phi_input_dim']}")
        print(f"[arch] critic_context_dim={c['critic_context_dim']}  "
              f"critic_extra_dim(z_onehot)={c['critic_extra_dim']}")
        print(f"[arch] actor_input_dim={c['actor_input_dim']}  "
              f"(local_obs + z_embed={c['actor_z_embed_dim']} + z_onehot={c['actor_z_onehot_dim']})")
        print(f"[arch] actor_z_residual_adapters={bool(c['actor_z_residual_enabled'])}  "
              f"latent_k={self.latent_k}")

    def set_sampling_generators(
        self,
        *,
        strategy: Optional[torch.Generator] = None,
        action: Optional[torch.Generator] = None,
    ) -> None:
        """Set dedicated RNGs for strategy vs. action sampling. ``None`` = PyTorch default (shared global) for that stream."""
        self._sampling_gen_strategy = strategy
        self._sampling_gen_action = action

    @staticmethod
    def _categorical_argmax_or_sample(
        dist: Categorical, *, deterministic: bool, generator: Optional[torch.Generator]
    ) -> torch.Tensor:
        if deterministic:
            return torch.argmax(dist.logits, dim=-1)
        if generator is not None:
            # ``Categorical.sample(generator=)`` is not available in all supported PyTorch versions;
            # ``torch.multinomial`` matches the same distribution and honors ``generator``.
            logits = dist.logits
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1, replacement=True, generator=generator).squeeze(-1)
        return dist.sample()

    def _forward_q_phi(
        self,
        global_state: torch.Tensor,
        selector_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        expected_context_dim = int(self.global_state_dim) if self.selector_gru is not None else int(self.q_phi_input_dim)
        if global_state.dim() != 2 or int(global_state.shape[1]) != expected_context_dim:
            raise ValueError(
                f"q_phi expected context shape (B, {expected_context_dim}), got {tuple(global_state.shape)}"
            )
        if self.strategy_encoder is None:
            raise RuntimeError("strategy encoder is not initialized.")
        if self.selector_gru is None:
            return self.strategy_encoder(global_state.float()), None
        if selector_hidden is None:
            raise RuntimeError("selector_hidden is required when the recurrent selector is enabled.")
        # GRU takes raw 34-dim state; encoder takes full (possibly augmented) state + hidden.
        gru_input = global_state[:, :GLOBAL_STATE_DIM].float()
        h_new = self.selector_gru(gru_input, selector_hidden)
        encoder_in = torch.cat([global_state.float(), h_new], dim=-1)
        return self.strategy_encoder(encoder_in), h_new

    @torch.no_grad()
    def advance_selector_hidden(
        self,
        global_state: torch.Tensor,
        selector_hidden: torch.Tensor,
        episode_boundary: torch.Tensor,
    ) -> torch.Tensor:
        """One GRU transition per env-step — V6I7 per-step hidden update.

        Called once after every env step for ALL environments (not just at
        decision steps). Caller passes ``episode_boundary = terminated | truncated``
        as a (B,) bool; done envs are zeroed after the GRU update.

        Returns detached (B, hidden_dim) tensor — no gradients.
        """
        if self.selector_gru is None:
            raise RuntimeError("advance_selector_hidden requires use_recurrent_selector=True")
        gs = global_state.float()
        if gs.dim() == 1:
            gs = gs.unsqueeze(0)
        if gs.shape[1] != self.global_state_dim:
            # Strip scheduler-phase column if it was appended (the GRU takes raw
            # global state, not the augmented 35-dim version).
            gs = gs[:, :GLOBAL_STATE_DIM]
        h_new = self.selector_gru(gs, selector_hidden.float())
        # Reset hidden state for environments that ended this step.
        if episode_boundary.any():
            mask = episode_boundary.to(device=h_new.device).float().unsqueeze(-1)
            h_new = h_new * (1.0 - mask)
        return h_new.detach()

    def forward_router_sequence(
        self,
        global_state_seq: torch.Tensor,
        h_start: torch.Tensor,
        done_mask_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """BPTT forward pass for the recurrent router over a contiguous chunk.

        Runs the GRU through all steps (with episode resets via done_mask_seq),
        then feeds [global_state, h_t] into q_phi at every step.  Gradients
        flow back through the full GRU unroll so the caller controls what to
        include in the loss (e.g. only loss-window steps, not burn-in).

        Args:
            global_state_seq: shape ``(T, B, state_dim)`` — 35-dim in V6I7
                (raw 34 + scheduler phase). The GRU strips the extra dim itself.
            h_start:          shape ``(B, hidden_dim)`` — starting hidden state,
                detached from prior rollout.
            done_mask_seq:    shape ``(T, B)`` — 1 where the episode ended AFTER
                this step (so h resets BEFORE the next step).

        Returns:
            logits:  ``(T, B, K)`` — q_phi logits at every chunk step.
            hiddens: ``(T, B, hidden_dim)`` — GRU hidden states, one per step,
                INCLUDING the reset applied after each done. Burn-in hiddens
                are still returned; the caller slices them off.
        """
        if self.selector_gru is None:
            raise RuntimeError("forward_router_sequence requires use_recurrent_selector=True")
        if self.strategy_encoder is None:
            raise RuntimeError("forward_router_sequence requires strategy_encoder")

        T, B, _ = global_state_seq.shape
        gru_in_seq = global_state_seq[:, :, :GLOBAL_STATE_DIM].float()

        h_t = h_start.float()
        all_logits: list[torch.Tensor] = []
        all_hiddens: list[torch.Tensor] = []

        for t in range(T):
            h_t = self.selector_gru(gru_in_seq[t], h_t)
            # Apply episode reset: any env that ended at t-1 gets zeroed hidden.
            if done_mask_seq[t].any():
                reset = done_mask_seq[t].float().unsqueeze(-1)
                h_t = h_t * (1.0 - reset)
            encoder_in = torch.cat([global_state_seq[t].float(), h_t], dim=-1)
            logits_t = self.strategy_encoder(encoder_in)
            all_logits.append(logits_t)
            all_hiddens.append(h_t)

        return torch.stack(all_logits, dim=0), torch.stack(all_hiddens, dim=0)

    def strategy_logits(
        self,
        global_state: torch.Tensor,
        *,
        selector_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``q_phi(z | s)`` logits for latent strategy mode.

        Since Step 5 the latent policy and the optional A2 aux-return head are
        backed by **separate** modules, so ``strategy_logits`` always reads
        ``self.strategy_encoder``. (Pre-Step-5, with the aux-return head on,
        the same module served both roles and the z-policy logits were
        ``strategy_aux_return_head(s) / strategy_tau``. Legacy checkpoints with
        that aliased layout get migrated by
        :func:`_migrate_legacy_aliased_strategy_modules` so the trained
        weights are mirrored into ``strategy_encoder``.)
        """
        if not self.uses_latent_strategy:
            raise RuntimeError("strategy_logits is only available when latent strategy is enabled.")
        logits, _ = self._forward_q_phi(global_state, selector_hidden)
        return logits / self.strategy_tau

    def _validate_z_idx(self, z_idx: torch.Tensor) -> torch.Tensor:
        return _validate_indices(z_idx, self.latent_k, "z_idx")

    def _build_selector_context(
        self,
        global_state: torch.Tensor,
        selector_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gs = global_state.float()
        if gs.dim() != 2 or int(gs.shape[1]) != int(self.global_state_dim):
            raise ValueError(
                f"selector context expected global_state shape (B, {self.global_state_dim}), "
                f"got {tuple(gs.shape)}"
            )
        if not self.use_recurrent_selector:
            return gs
        if selector_hidden is None:
            raise ValueError("selector_hidden is required when the recurrent selector is enabled.")
        hidden = selector_hidden.float()
        expected_hidden = (int(gs.shape[0]), int(self.recurrent_selector_hidden_dim))
        if tuple(hidden.shape) != expected_hidden:
            raise ValueError(
                f"selector_hidden must have shape {expected_hidden}, got {tuple(hidden.shape)}"
            )
        return torch.cat([gs, hidden], dim=-1)

    def strategy_aux_return_predictions(self, global_state: torch.Tensor) -> torch.Tensor:
        """A2 auxiliary: per-z scalar predictions from the shared trunk, shape ``(B, K)``.

        These are **not** a full action-value :math:`Q(s,\\mathbf{a}, z)` and are not trained with
        off-policy Bellman targets; they only supply an optional supervised signal on the **sampled**
        strategy index (see plan A2 / auxiliary return regression).
        """
        if not self.uses_latent_strategy or self.strategy_aux_return_head is None:
            raise RuntimeError(
                "strategy_aux_return_predictions is only available when the A2 auxiliary return head is enabled."
            )
        return self.strategy_aux_return_head(global_state.float())

    def episode_strategy_value(
        self,
        global_state: torch.Tensor,
        z_idx: torch.Tensor,
        *,
        selector_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Episode-level baseline V_phi(context, z) for router PPO credit."""
        if not self.uses_latent_strategy or self.episode_strategy_value_head is None:
            raise RuntimeError("episode_strategy_value is only available for episode-level strategy PPO.")
        context = self._build_selector_context(global_state, selector_hidden)
        if context.dim() != 2 or int(context.shape[1]) != int(self.q_phi_input_dim):
            raise ValueError(
                f"episode strategy value expected context shape (B, {self.q_phi_input_dim}), "
                f"got {tuple(context.shape)}"
            )
        z = self._validate_z_idx(z_idx)
        if int(z.shape[0]) != int(context.shape[0]):
            raise ValueError(f"z_idx must have shape ({int(context.shape[0])},), got {tuple(z_idx.shape)}")
        z_one_hot = F.one_hot(z, num_classes=self.latent_k).to(dtype=torch.float32, device=context.device)
        return self.episode_strategy_value_head(torch.cat([context, z_one_hot], dim=-1)).squeeze(-1)

    def sample_strategy(
        self,
        global_state: torch.Tensor,
        *,
        deterministic: bool = False,
        selector_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Sample or greedily choose team strategy indices from ``q_phi(z | s)``."""
        logits, h_new = self._forward_q_phi(global_state, selector_hidden)
        tempered = logits / self.strategy_tau
        dist = Categorical(logits=tempered)
        z_idx = self._categorical_argmax_or_sample(
            dist, deterministic=deterministic, generator=self._sampling_gen_strategy
        )
        return z_idx.long(), dist.log_prob(z_idx), dist.entropy(), tempered, h_new

    def phase_logits_from_strategy_logits(self, z_logits: torch.Tensor) -> torch.Tensor:
        """Predict team phase through q_phi's soft z distribution."""
        if not self.uses_latent_strategy or self.strategy_embedding is None or self.phase_predictor is None:
            raise RuntimeError("phase logits are only available when latent strategy is enabled.")
        if z_logits.dim() != 2 or int(z_logits.shape[1]) != int(self.latent_k):
            raise ValueError(
                f"phase predictor expected z logits shape (B, {self.latent_k}), got {tuple(z_logits.shape)}"
            )
        z_probs = torch.softmax(z_logits.float(), dim=-1)
        expected_z_emb = z_probs @ self.strategy_embedding.weight
        return self.phase_predictor(expected_z_emb)

    def _encode_local_obs(
        self,
        obs: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(local_in, cnn_features, agent_mask)`` with shape ``(B*N, F)``."""
        grid = obs["grid"].float()
        vec = obs["vec"].float()
        if grid.dim() != 5:
            raise ValueError(f"grid must have shape (B, N, C, H, W), got {tuple(grid.shape)}")
        if vec.dim() != 3:
            raise ValueError(f"vec must have shape (B, N, V), got {tuple(vec.shape)}")
        batch = int(grid.shape[0])
        if int(grid.shape[1]) != self.n_agents or tuple(int(v) for v in grid.shape[2:]) != self.grid_shape:
            raise ValueError(
                f"grid must have shape (B, {self.n_agents}, {self.grid_shape[0]}, "
                f"{self.grid_shape[1]}, {self.grid_shape[2]}), got {tuple(grid.shape)}"
            )
        if int(vec.shape[1]) != self.n_agents or int(vec.shape[2]) != self.vec_dim:
            raise ValueError(f"vec must have shape (B, {self.n_agents}, {self.vec_dim}), got {tuple(vec.shape)}")
        cnn_features = self.actor_cnn(grid.reshape(batch * self.n_agents, *self.grid_shape))
        cnn_features = cnn_features.reshape(batch, self.n_agents, self.actor_cnn_feature_dim)
        vloc = vec.float()
        agent_mask = obs.get("agent_mask")
        if agent_mask is not None:
            if agent_mask.dim() == 1:
                agent_mask = agent_mask.unsqueeze(0)
            mask = agent_mask.float().unsqueeze(-1)
            cnn_features = cnn_features * mask
            vloc = vloc * mask
        else:
            mask = torch.ones((batch, self.n_agents, 1), dtype=torch.float32, device=grid.device)
        local_obs = torch.cat([cnn_features, vloc], dim=-1)
        local_in = local_obs.reshape(batch * self.n_agents, -1)
        if int(local_in.shape[-1]) != int(self._local_actor_in_dim):
            raise AssertionError(
                f"local actor input width {int(local_in.shape[-1])} != expected "
                f"{int(self._local_actor_in_dim)}"
            )
        return local_in, cnn_features, mask.squeeze(-1)

    def _message_head_input(
        self,
        local_in: torch.Tensor,
        *,
        batch: int,
        z_idx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.communication_enabled or self.message_head is None:
            raise RuntimeError("message head requested but communication is disabled")
        features = local_in.reshape(batch, self.n_agents, -1)
        if self.uses_latent_strategy:
            if z_idx is None:
                raise ValueError("z_idx is required for message head when latent strategy is enabled.")
            z = self._validate_z_idx(z_idx)
            z_per_agent = z.unsqueeze(1).expand(batch, self.n_agents)
            z_emb = self.latent_actor.strategy_embedding(z_per_agent.reshape(-1))
            features = torch.cat([features, z_emb.reshape(batch, self.n_agents, -1)], dim=-1)
        flat = features.reshape(batch * self.n_agents, -1)
        if int(flat.shape[-1]) != int(self._message_head_in_dim):
            raise AssertionError(
                f"message head input width {int(flat.shape[-1])} != expected {self._message_head_in_dim}"
            )
        return flat

    def message_logits(
        self,
        obs: Dict[str, torch.Tensor],
        *,
        z_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-agent message logits with shape ``(B, N, comm_num_symbols)``."""
        local_in, _, _ = self._encode_local_obs(obs)
        batch = int(obs["grid"].shape[0])
        head_in = self._message_head_input(local_in, batch=batch, z_idx=z_idx)
        logits = self.message_head(head_in).reshape(batch, self.n_agents, int(self.comm_num_symbols))
        return logits

    def _sample_messages(
        self,
        obs: Dict[str, torch.Tensor],
        *,
        z_idx: Optional[torch.Tensor],
        comm_boundary_mask: Optional[torch.Tensor],
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Sample outbound symbols and PPO log-probs for one env step.

        Message PPO credit is **boundary-only**: held symbols persist in the
        transport for ``comm_interval_steps`` decision steps, but
        ``message_log_probs`` / ``message_entropy`` are nonzero only when
        ``comm_boundary_mask`` is true. Non-boundary rows must not duplicate
        policy loss for the same held symbol.
        """
        if not self.communication_enabled:
            batch = int(obs["grid"].shape[0])
            device = obs["grid"].device
            return {
                "message_symbols": torch.zeros((batch, self.n_agents), dtype=torch.long, device=device),
                "message_log_probs": torch.zeros((batch,), dtype=torch.float32, device=device),
                "message_entropy": torch.zeros((batch,), dtype=torch.float32, device=device),
                "message_boundary_mask": torch.zeros((batch,), dtype=torch.bool, device=device),
            }
        logits = self.message_logits(obs, z_idx=z_idx)
        batch = int(logits.shape[0])
        device = logits.device
        boundary = (
            comm_boundary_mask.bool().reshape(batch)
            if comm_boundary_mask is not None
            else torch.zeros((batch,), dtype=torch.bool, device=device)
        )
        dist = Categorical(logits=logits.reshape(batch * self.n_agents, -1))
        flat_logits = logits.reshape(batch * self.n_agents, -1)
        flat_dist = Categorical(logits=flat_logits)
        if deterministic:
            symbols = flat_dist.probs.argmax(dim=-1)
        else:
            g_act = self._sampling_gen_action
            symbols = self._categorical_argmax_or_sample(
                flat_dist, deterministic=False, generator=g_act
            )
        symbols = symbols.reshape(batch, self.n_agents)
        per_agent_logprob = flat_dist.log_prob(symbols.reshape(-1)).reshape(batch, self.n_agents)
        per_agent_entropy = flat_dist.entropy().reshape(batch, self.n_agents)
        alive = obs.get("agent_mask")
        if alive is not None:
            alive_mask = alive.float()
            if alive_mask.dim() == 1:
                alive_mask = alive_mask.unsqueeze(0)
            per_agent_logprob = per_agent_logprob * alive_mask
            per_agent_entropy = per_agent_entropy * alive_mask
        message_log_probs = per_agent_logprob.sum(dim=-1)
        message_entropy = per_agent_entropy.sum(dim=-1)
        boundary_f = boundary.to(dtype=torch.float32, device=device)
        return {
            "message_symbols": symbols,
            "message_log_probs": message_log_probs * boundary_f,
            "message_entropy": message_entropy * boundary_f,
            "message_boundary_mask": boundary,
        }

    def _evaluate_messages(
        self,
        obs: Dict[str, torch.Tensor],
        *,
        message_symbols: torch.Tensor,
        message_boundary_mask: torch.Tensor,
        z_idx: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.communication_enabled:
            batch = int(obs["grid"].shape[0])
            device = obs["grid"].device
            return (
                torch.zeros((batch,), dtype=torch.float32, device=device),
                torch.zeros((batch,), dtype=torch.float32, device=device),
            )
        batch = int(obs["grid"].shape[0])
        device = obs["grid"].device
        boundary = message_boundary_mask.bool().reshape(batch)
        zero = torch.zeros((batch,), dtype=torch.float32, device=device)
        if not bool(boundary.any()):
            return zero, zero

        active_idx = torch.where(boundary)[0]
        obs_active: Dict[str, torch.Tensor] = {}
        for key, value in obs.items():
            if isinstance(value, torch.Tensor) and int(value.shape[0]) == batch:
                obs_active[key] = value.index_select(0, active_idx)
            else:
                obs_active[key] = value
        symbols_active = message_symbols.long().index_select(0, active_idx)
        z_active = None
        if z_idx is not None:
            z_active = z_idx.index_select(0, active_idx)

        logits = self.message_logits(obs_active, z_idx=z_active)
        n_active = int(active_idx.shape[0])
        dist = Categorical(logits=logits.reshape(n_active * self.n_agents, -1))
        per_agent_logprob = dist.log_prob(symbols_active.reshape(-1)).reshape(
            n_active, self.n_agents
        )
        per_agent_entropy = dist.entropy().reshape(n_active, self.n_agents)
        alive = obs_active.get("agent_mask")
        if alive is not None:
            alive_mask = alive.float()
            if alive_mask.dim() == 1:
                alive_mask = alive_mask.unsqueeze(0)
            per_agent_logprob = per_agent_logprob * alive_mask
            per_agent_entropy = per_agent_entropy * alive_mask
        active_log_probs = per_agent_logprob.sum(dim=-1)
        active_entropy = per_agent_entropy.sum(dim=-1)
        message_log_probs = zero.scatter(0, active_idx, active_log_probs)
        message_entropy = zero.scatter(0, active_idx, active_entropy)
        return message_log_probs, message_entropy

    def policy_logits(
        self,
        obs: Dict[str, torch.Tensor],
        z_idx: Optional[torch.Tensor] = None,
        *,
        detach_local_features: bool = False,
    ) -> torch.Tensor:
        """Return flattened MultiDiscrete logits with shape ``(B, sum(action_dims))``.

        Feature pipeline:

        1. ``actor_cnn`` encodes per-agent grids → ``cnn_features``.
        2. Optional agent mask zeroes out padded agents' features / scalars.
        3. ``local_features = concat(cnn_features, scalars)`` per-agent.
        4. ``self.latent_actor`` handles the strategy embedding (when present)
           and the 256-256 MLP + action head. Per-agent ``z`` is shared across
           the team — the same ``z_idx`` row is broadcast across all agents.
        """
        local_in, _, _ = self._encode_local_obs(obs)
        if detach_local_features:
            local_in = local_in.detach()
        batch = int(obs["grid"].shape[0])
        if self.uses_latent_strategy:
            if z_idx is None:
                raise ValueError("z_idx is required when latent strategy is enabled.")
            z = self._validate_z_idx(z_idx)
            if z.shape[0] != batch:
                raise ValueError(f"z_idx must have shape ({batch},), got {tuple(z_idx.shape)}")
            z_per_agent = z.unsqueeze(1).expand(batch, self.n_agents).reshape(batch * self.n_agents)
            per_agent_flat = self.latent_actor(local_in, z_per_agent)
        else:
            per_agent_flat = self.latent_actor(local_in)
        per_agent_logits = per_agent_flat.reshape(batch, self.n_agents, self.per_agent_logits)
        if int(per_agent_logits.shape[-1]) == 0:
            raise AssertionError("latent_actor produced zero-width logits; check action_dim wiring")
        return per_agent_logits.reshape(batch, self.n_agents * self.per_agent_logits)

    def policy_trunk_features(
        self, obs: Dict[str, torch.Tensor], z_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return per-agent trunk features with shape ``(B, n_agents, hidden_dim)``."""
        local_in, _, _ = self._encode_local_obs(obs)
        batch = int(obs["grid"].shape[0])
        if self.uses_latent_strategy:
            if z_idx is None:
                raise ValueError("z_idx is required when latent strategy is enabled.")
            z = self._validate_z_idx(z_idx)
            if z.shape[0] != batch:
                raise ValueError(f"z_idx must have shape ({batch},), got {tuple(z_idx.shape)}")
            z_per_agent = z.unsqueeze(1).expand(batch, self.n_agents).reshape(batch * self.n_agents)
            hidden = self.latent_actor.trunk_features(local_in, z_per_agent)
        else:
            hidden = self.latent_actor.trunk_features(local_in)
        return hidden.reshape(batch, self.n_agents, int(self.latent_actor.hidden_dim))

    def _joint_action_one_hot(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.long()
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        if actions.shape[1] != len(self.action_dims):
            raise ValueError(
                f"actions must have shape (B, {len(self.action_dims)}), got {tuple(actions.shape)}"
            )
        chunks = []
        for col, dim in enumerate(self.action_dims):
            action = _validate_indices(actions[:, col], int(dim), f"actions[:, {col}]")
            chunks.append(F.one_hot(action, num_classes=int(dim)).float())
        return torch.cat(chunks, dim=-1)

    def _critic_extra(self, z_idx: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.uses_latent_strategy:
            return None
        if z_idx is None:
            raise ValueError("z_idx is required for critic conditioning in latent strategy mode.")
        z = self._validate_z_idx(z_idx)
        return F.one_hot(z, num_classes=self.latent_k).float()

    def values(
        self,
        global_state: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        z_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return scalar :math:`V_\\phi(s, z)` with shape ``(B,)`` (PPO/GAE baseline)."""
        if global_state.dim() != 2 or int(global_state.shape[1]) != int(self.critic_context_dim):
            raise ValueError(
                f"critic expected context shape (B, {self.critic_context_dim}), got {tuple(global_state.shape)}"
            )
        if self.uses_latent_strategy and actions is not None:
            raise ValueError(
                "The PPO value critic is conditioned on z only; pass z_idx and omit actions."
            )
        return self.critic(global_state.float(), extra=self._critic_extra(z_idx)).squeeze(-1)

    def _mask_logits(self, logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return logits
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        mask = mask.float()
        if mask.shape != logits.shape:
            raise ValueError(
                f"action mask must have shape {tuple(logits.shape)}, got {tuple(mask.shape)}"
            )
        masked_chunks = []
        offset = 0
        for dim in self.action_dims:
            chunk = logits[:, offset : offset + dim]
            mask_chunk = mask[:, offset : offset + dim]
            all_invalid = mask_chunk.sum(dim=1, keepdim=True) <= 0.0
            if bool(all_invalid.any().item()):
                raise ValueError("action mask has a head with no valid actions")
            masked_chunks.append(chunk.masked_fill(mask_chunk <= 0.0, -1e8))
            offset += dim
        return torch.cat(masked_chunks, dim=1)

    def _categoricals(self, logits: torch.Tensor) -> Iterable[Categorical]:
        offset = 0
        for dim in self.action_dims:
            yield Categorical(logits=logits[:, offset : offset + dim])
            offset += dim

    def _log_prob_entropy(self, logits: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions = actions.long()
        log_probs = []
        entropies = []
        for col, dist in enumerate(self._categoricals(logits)):
            action = _validate_indices(actions[:, col], int(dist.logits.shape[1]), f"actions[:, {col}]")
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
        return torch.stack(log_probs, dim=0).sum(dim=0), torch.stack(entropies, dim=0).sum(dim=0)

    def act(
        self,
        obs: Dict[str, torch.Tensor],
        global_state: torch.Tensor,
        *,
        deterministic: bool = False,
        z_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or greedily select actions and return values/log-probs/entropy."""
        if self.uses_latent_strategy and z_idx is None:
            raise ValueError("Sample and provide z_idx before calling act() when latent strategy is enabled.")
        logits = self._mask_logits(self.policy_logits(obs, z_idx=z_idx), obs.get("mask"))
        actions = []
        g_act = self._sampling_gen_action
        for dist in self._categoricals(logits):
            actions.append(
                self._categorical_argmax_or_sample(
                    dist, deterministic=deterministic, generator=g_act
                )
            )
        action_tensor = torch.stack(actions, dim=1)
        log_prob, entropy = self._log_prob_entropy(logits, action_tensor)
        values = self.values(global_state, z_idx=z_idx)
        return action_tensor, values, log_prob, entropy

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        global_state: torch.Tensor,
        actions: torch.Tensor,
        *,
        z_idx: Optional[torch.Tensor] = None,
        selector_hidden: Optional[torch.Tensor] = None,
        router_context: Optional[torch.Tensor] = None,
        message_symbols: Optional[torch.Tensor] = None,
        message_boundary_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate fixed actions under the current policy."""
        logits = self._mask_logits(self.policy_logits(obs, z_idx=z_idx), obs.get("mask"))
        log_prob, entropy = self._log_prob_entropy(logits, actions)
        values = self.values(global_state, z_idx=z_idx)
        aux: dict[str, torch.Tensor] = {}
        if self.communication_enabled and message_symbols is not None and message_boundary_mask is not None:
            msg_log_prob, msg_entropy = self._evaluate_messages(
                obs,
                message_symbols=message_symbols,
                message_boundary_mask=message_boundary_mask,
                z_idx=z_idx,
            )
            aux["message_log_probs"] = msg_log_prob
            aux["message_entropy"] = msg_entropy
        if self.uses_latent_strategy:
            if z_idx is None:
                raise ValueError("z_idx is required when latent strategy is enabled.")
            q_context = router_context if router_context is not None else global_state
            if self.use_recurrent_selector:
                if selector_hidden is None:
                    raise ValueError(
                        "selector_hidden from rollout collection is required "
                        "when evaluating recurrent strategy actions."
                    )
                z_logits = self.strategy_logits(q_context, selector_hidden=selector_hidden)
            else:
                z_logits = self.strategy_logits(q_context)
            z_dist = Categorical(logits=z_logits)
            z = self._validate_z_idx(z_idx)
            aux["strategy_logits"] = z_logits
            aux["strategy_log_prob"] = z_dist.log_prob(z)
            aux["strategy_entropy"] = z_dist.entropy()
        return values, log_prob, entropy, aux
