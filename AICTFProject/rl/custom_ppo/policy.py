from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from rl.global_state import GLOBAL_STATE_DIM
from rl.latent_marl import LatentConditionedActor, StrategyEncoder, CONTEXT_STATE_DIM
from rl.latent_phase_labels import TEAM_PHASES
from rl.networks import CNNEncoder, CentralizedCritic


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
        # New-layout checkpoint already has both heads; if the model expects no
        # aux head we'd drop those keys below, but state_dict load handles that.
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
        strategy_tau: float = 1.0,
        latent_actor_z_onehot_enabled: bool = False,
        latent_actor_z_onehot_scale: float = 1.0,
        latent_actor_z_embed_scale: float = 1.0,
        latent_actor_z_adapter_enabled: bool = False,
        latent_actor_z_adapter_scale: float = 0.0,
        latent_actor_z_adapter_init_std: float = 0.02,
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
        self.strategy_tau = max(1e-3, float(strategy_tau))

        self.global_state_dim = CONTEXT_STATE_DIM if self.uses_latent_strategy else GLOBAL_STATE_DIM

        if self.uses_latent_strategy:
            # Step 5: ``StrategyEncoder`` (q_phi(z|s), the latent team-strategy policy)
            # and the optional A2 auxiliary per-z return regression head are now
            # ALWAYS distinct ``nn.Module`` instances when both are enabled. Before
            # this change a single ``StrategyEncoder`` was aliased to either slot
            # depending on the cfg flag, which made the same code path mean
            # different things at runtime ("q_phi is the aux head, sometimes").
            # See ``_migrate_legacy_aliased_strategy_modules`` for the on-disk
            # checkpoint migration.
            self.strategy_encoder = StrategyEncoder(
                state_dim=self.global_state_dim,
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
        )
        critic_extra_dim = self.joint_action_onehot_dim + self.latent_k if self.uses_latent_strategy else 0
        self.critic = CentralizedCritic(
            global_state_dim=self.global_state_dim,
            hidden_dim=int(critic_hidden_dim),
            extra_dim=critic_extra_dim,
        )
        if self.use_episode_strategy_value_head:
            episode_value_in = int(self.global_state_dim + self.latent_k)
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
        self.critic_joint_action_dim = int(self.joint_action_onehot_dim) if self.uses_latent_strategy else 0
        self.actor_input_dim = int(self._decentralized_actor_in_dim)
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
            assert int(self.global_state_dim) == int(CONTEXT_STATE_DIM), (
                f"latent global_state_dim must be {CONTEXT_STATE_DIM}, got {self.global_state_dim}"
            )
            assert int(self.q_phi_input_dim) == int(CONTEXT_STATE_DIM), (
                f"q_phi_input_dim must be {CONTEXT_STATE_DIM}, got {self.q_phi_input_dim}"
            )
            assert int(self.critic.global_state_dim) == int(CONTEXT_STATE_DIM), (
                f"critic global_state_dim must be {CONTEXT_STATE_DIM}, got {self.critic.global_state_dim}"
            )
            assert int(self._decentralized_actor_in_dim) == actor_expected, f"latent actor input dim must be {actor_expected}, got {self._decentralized_actor_in_dim}"
            expected_extra = int(self.joint_action_onehot_dim + self.latent_k)
            assert int(self.critic.extra_dim) == expected_extra, (
                f"critic extra_dim must be joint_action_onehot_dim + latent_k = {expected_extra}, got {self.critic.extra_dim}"
            )
        else:
            assert int(self.global_state_dim) == int(GLOBAL_STATE_DIM), (
                f"no-latent global_state_dim must be {GLOBAL_STATE_DIM}, got {self.global_state_dim}"
            )
            assert int(self.critic.global_state_dim) == int(GLOBAL_STATE_DIM), (
                f"no-latent critic global_state_dim must be {GLOBAL_STATE_DIM}, got {self.critic.global_state_dim}"
            )
            assert int(self._decentralized_actor_in_dim) == 148, f"no-latent actor input dim must be 148, got {self._decentralized_actor_in_dim}"
            assert int(self.critic.extra_dim) == 0, f"no-latent critic extra_dim must be 0, got {self.critic.extra_dim}"

        if int(self._decentralized_actor_in_dim) != actor_expected:
            raise AssertionError(
                f"actor_input_dim={self._decentralized_actor_in_dim} must equal local obs + z embedding width "
                f"{actor_expected}"
            )
        first_actor = self.actor_body[0]
        if not isinstance(first_actor, nn.Linear) or int(first_actor.in_features) != actor_expected:
            got = getattr(first_actor, "in_features", None)
            raise AssertionError(f"actor MLP first layer input {got} != decentralized actor input {actor_expected}")
        if int(self._decentralized_actor_in_dim) == int(CONTEXT_STATE_DIM):
            raise AssertionError(
                f"actor_input_dim={self._decentralized_actor_in_dim} equals temporal_context_dim={CONTEXT_STATE_DIM}; "
                "actor must consume local obs + z embedding only, never the centralized temporal context."
            )

    def input_dim_contract(self) -> dict[str, int]:
        self._assert_input_contracts()
        return {
            "base_global_state_dim": int(GLOBAL_STATE_DIM),
            "temporal_context_dim": int(CONTEXT_STATE_DIM),
            "q_phi_input_dim": int(self.q_phi_input_dim),
            "critic_context_dim": int(self.critic_context_dim),
            "actor_input_dim": int(self.actor_input_dim),
            "actor_z_embed_dim": int(self.z_embed_dim),
            "actor_z_onehot_dim": int(self.z_onehot_dim),
            "critic_extra_dim": int(self.critic.extra_dim),
            "critic_z_dim": int(self.critic_z_dim),
            "critic_joint_action_dim": int(self.critic_joint_action_dim),
        }

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

    def strategy_logits(self, global_state: torch.Tensor) -> torch.Tensor:
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
        if global_state.dim() != 2 or int(global_state.shape[1]) != int(self.q_phi_input_dim):
            raise AssertionError(
                f"q_phi expected context shape (B, {self.q_phi_input_dim}), got {tuple(global_state.shape)}"
            )
        if self.strategy_encoder is None:
            raise RuntimeError("strategy encoder is not initialized.")
        return self.strategy_encoder(global_state.float())

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

    def episode_strategy_value(self, global_state: torch.Tensor, z_idx: torch.Tensor) -> torch.Tensor:
        """Episode-level baseline V_phi(s0, z) for PPO credit on q_phi's sampled strategy action."""
        if not self.uses_latent_strategy or self.episode_strategy_value_head is None:
            raise RuntimeError("episode_strategy_value is only available for episode-level strategy PPO.")
        if global_state.dim() != 2 or int(global_state.shape[1]) != int(self.q_phi_input_dim):
            raise AssertionError(
                f"episode strategy value expected context shape (B, {self.q_phi_input_dim}), got {tuple(global_state.shape)}"
            )
        z = z_idx.long().reshape(-1).clamp(min=0, max=self.latent_k - 1)
        if int(z.shape[0]) != int(global_state.shape[0]):
            raise ValueError(f"z_idx must have shape ({int(global_state.shape[0])},), got {tuple(z_idx.shape)}")
        z_one_hot = F.one_hot(z, num_classes=self.latent_k).to(dtype=torch.float32, device=global_state.device)
        return self.episode_strategy_value_head(torch.cat([global_state.float(), z_one_hot], dim=-1)).squeeze(-1)

    def sample_strategy(
        self,
        global_state: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or greedily choose team strategy indices from ``q_phi(z | s)``."""
        logits = self.strategy_logits(global_state)
        dist = Categorical(logits=logits)
        z_idx = self._categorical_argmax_or_sample(
            dist, deterministic=deterministic, generator=self._sampling_gen_strategy
        )
        return z_idx.long(), dist.log_prob(z_idx), dist.entropy(), logits

    def phase_logits_from_strategy_logits(self, z_logits: torch.Tensor) -> torch.Tensor:
        """Predict team phase through q_phi's soft z distribution."""
        if not self.uses_latent_strategy or self.strategy_embedding is None or self.phase_predictor is None:
            raise RuntimeError("phase logits are only available when latent strategy is enabled.")
        if z_logits.dim() != 2 or int(z_logits.shape[1]) != int(self.latent_k):
            raise AssertionError(f"phase predictor expected z logits shape (B, {self.latent_k}), got {tuple(z_logits.shape)}")
        z_probs = torch.softmax(z_logits.float(), dim=-1)
        expected_z_emb = z_probs @ self.strategy_embedding.weight
        return self.phase_predictor(expected_z_emb)

    def policy_logits(self, obs: Dict[str, torch.Tensor], z_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return flattened MultiDiscrete logits with shape ``(B, sum(action_dims))``.

        Feature pipeline:

        1. ``actor_cnn`` encodes per-agent grids → ``cnn_features``.
        2. Optional agent mask zeroes out padded agents' features / scalars.
        3. ``local_features = concat(cnn_features, scalars)`` per-agent.
        4. ``self.latent_actor`` handles the strategy embedding (when present)
           and the 256-256 MLP + action head. Per-agent ``z`` is shared across
           the team — the same ``z_idx`` row is broadcast across all agents.
        """
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
        local_obs = torch.cat([cnn_features, vloc], dim=-1)
        local_in = local_obs.reshape(batch * self.n_agents, -1)
        if int(local_in.shape[-1]) != int(self._local_actor_in_dim):
            raise AssertionError(
                f"local actor input width {int(local_in.shape[-1])} != expected "
                f"{int(self._local_actor_in_dim)}"
            )
        if self.uses_latent_strategy:
            if z_idx is None:
                raise ValueError("z_idx is required when latent strategy is enabled.")
            z = z_idx.long().reshape(-1).clamp(min=0, max=self.latent_k - 1)
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

    def _joint_action_one_hot(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.long()
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        chunks = []
        for col, dim in enumerate(self.action_dims):
            action = actions[:, col].clamp(min=0, max=dim - 1)
            chunks.append(F.one_hot(action, num_classes=dim).float())
        return torch.cat(chunks, dim=-1)

    def _critic_extra(self, actions: Optional[torch.Tensor], z_idx: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.uses_latent_strategy:
            return None
        assert z_idx is not None, "z_idx is required for critic conditioning in latent strategy mode"
        if actions is None:
            raise ValueError("actions are required by the latent action-conditioned **value** critic.")
        z = z_idx.long().reshape(-1).clamp(min=0, max=self.latent_k - 1)
        z_one_hot = F.one_hot(z, num_classes=self.latent_k).float()
        extra = torch.cat([self._joint_action_one_hot(actions).to(z_one_hot.device), z_one_hot], dim=-1)
        expected = int(self.joint_action_onehot_dim + self.latent_k)
        if extra.dim() != 2 or int(extra.shape[1]) != expected:
            raise AssertionError(f"critic extra must be joint_action_onehot + z_onehot width {expected}, got {tuple(extra.shape)}")
        z_slice = extra[:, -self.latent_k :]
        z_sum = z_slice.sum(dim=-1)
        if int(z_slice.shape[1]) != int(self.latent_k) or not torch.allclose(z_sum, torch.ones_like(z_sum), atol=1e-6):
            raise AssertionError("critic input is missing the terminal z one-hot slice")
        return extra

    def values(
        self,
        global_state: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        z_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return scalar :math:`V_\\phi(s,\\mathbf{a},z)` with shape ``(B,)`` (PPO/GAE target)."""
        if global_state.dim() != 2 or int(global_state.shape[1]) != int(self.critic_context_dim):
            raise AssertionError(
                f"critic expected context shape (B, {self.critic_context_dim}), got {tuple(global_state.shape)}"
            )
        return self.critic(global_state.float(), extra=self._critic_extra(actions, z_idx)).squeeze(-1)

    def _mask_logits(self, logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return logits
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        mask = mask.float()
        masked_chunks = []
        offset = 0
        for dim in self.action_dims:
            chunk = logits[:, offset : offset + dim]
            mask_chunk = mask[:, offset : offset + dim]
            if mask_chunk.shape[1] < dim:
                pad = torch.ones((mask.shape[0], dim - mask_chunk.shape[1]), device=mask.device)
                mask_chunk = torch.cat([mask_chunk, pad], dim=1)
            all_invalid = mask_chunk.sum(dim=1, keepdim=True) <= 0.0
            safe_mask = torch.where(all_invalid, torch.ones_like(mask_chunk), mask_chunk)
            masked_chunks.append(chunk.masked_fill(safe_mask <= 0.0, -1e8))
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
            action = actions[:, col].clamp(min=0, max=dist.logits.shape[1] - 1)
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
            z_idx, _, _, _ = self.sample_strategy(global_state, deterministic=deterministic)
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
        values = self.values(global_state, actions=action_tensor, z_idx=z_idx)
        return action_tensor, values, log_prob, entropy

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        global_state: torch.Tensor,
        actions: torch.Tensor,
        *,
        z_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate fixed actions under the current policy."""
        logits = self._mask_logits(self.policy_logits(obs, z_idx=z_idx), obs.get("mask"))
        log_prob, entropy = self._log_prob_entropy(logits, actions)
        values = self.values(global_state, actions=actions, z_idx=z_idx)
        aux: dict[str, torch.Tensor] = {}
        if self.uses_latent_strategy:
            if z_idx is None:
                raise ValueError("z_idx is required when latent strategy is enabled.")
            z_logits = self.strategy_logits(global_state)
            z_dist = Categorical(logits=z_logits)
            z = z_idx.long().reshape(-1).clamp(min=0, max=self.latent_k - 1)
            aux["strategy_logits"] = z_logits
            aux["strategy_log_prob"] = z_dist.log_prob(z)
            aux["strategy_entropy"] = z_dist.entropy()
        return values, log_prob, entropy, aux
