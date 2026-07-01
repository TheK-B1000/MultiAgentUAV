"""Contiguous-chunk sequence minibatch sampler for V6I7 GRU router (BPTT).

Architecture note
-----------------
The actor PPO loop continues to use randomly shuffled transition-level
minibatches (``TensorDictRolloutBuffer.iter_minibatches(shuffle=True)``).
Only the router update uses this module — each chunk carries a starting
hidden state and is processed sequentially so gradients flow back through
the GRU via truncated BPTT.

Chunk layout
------------
Total chunk length  = ``burn_in`` + ``seq_len`` steps.
  * Burn-in prefix (first ``burn_in`` steps): forward the GRU to warm up
    hidden state; no loss computed, no gradients accumulated.
  * Loss window (last ``seq_len`` steps): all router losses computed here.

Cross-env independence
----------------------
Chunks are drawn independently per environment. An episode boundary
(done = terminated | truncated) inside a chunk causes the GRU hidden
to reset to zero at that step; this is handled by the caller during the
BPTT forward pass using the ``done_mask`` field in each chunk.
"""

from __future__ import annotations

from typing import Iterator

import torch

from rl.ppo_core import TensorDictRolloutBuffer

# Integer codes stored in buffer field "interval_end_reason".
REASON_MID_HOLD = 0
REASON_NEXT_OPPORTUNITY = 1
REASON_TERMINATED = 2
REASON_TRUNCATED = 3
REASON_BUFFER_CUT = 4


def iter_router_sequence_minibatches(
    buffer: TensorDictRolloutBuffer,
    *,
    burn_in: int,
    seq_len: int,
    chunks_per_batch: int,
    required_fields: tuple[str, ...] | None = None,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield contiguous sequence minibatches from a filled rollout buffer.

    Parameters
    ----------
    buffer:
        A fully filled ``TensorDictRolloutBuffer`` (``buffer.pos == buffer.buffer_size``).
    burn_in:
        Number of leading steps used to warm up GRU hidden state.
        No loss is computed over these steps.
    seq_len:
        Number of loss-bearing steps per chunk.
    chunks_per_batch:
        How many independent sequences per yielded batch.
    required_fields:
        If given, raise if any of these fields are missing from the buffer.
        Useful to assert that V6I7 fields are present before training.
    shuffle:
        Shuffle the chunk order across batches.
    generator:
        Optional RNG for reproducible shuffling.

    Yields
    ------
    dict[str, torch.Tensor]
        Each value has shape ``[chunk_total, chunks_per_batch, *field_shape]``
        where ``chunk_total = burn_in + seq_len``.  The first ``burn_in``
        steps along dim-0 are the warm-up prefix.

        The ``"selector_hidden_start"`` key is added by this function — it
        carries the stored GRU hidden state at the start of each chunk with
        shape ``[chunks_per_batch, hidden_dim]``.  If ``selector_hidden`` is
        not registered in the buffer (non-recurrent baseline) this key is
        absent.
    """
    if required_fields:
        missing = [f for f in required_fields if f not in buffer.fields]
        if missing:
            raise KeyError(
                f"iter_router_sequence_minibatches: required fields missing "
                f"from rollout buffer: {missing}"
            )

    T = int(buffer.pos)
    B = int(buffer.n_envs)
    chunk_total = int(burn_in) + int(seq_len)
    device = buffer.device

    if T < chunk_total:
        return  # buffer too short to form even one chunk

    # --- Enumerate all valid (start_t, env_b) chunk origins ---------------
    n_chunks_per_env = T // chunk_total
    if n_chunks_per_env == 0:
        return

    # All chunk starts are aligned to multiples of chunk_total per env.
    # Shape: (n_chunks_per_env * B,)  — each row is (start_t, env_b).
    start_ts = torch.arange(0, n_chunks_per_env * chunk_total, chunk_total, device=device)
    env_ids = torch.arange(B, device=device)
    # Cartesian product: all (start_t, env_b) pairs.
    grid_t = start_ts.repeat_interleave(B)  # (n_chunks_per_env * B,)
    grid_e = env_ids.repeat(n_chunks_per_env)  # (n_chunks_per_env * B,)
    n_chunks_total = int(grid_t.numel())

    if n_chunks_total == 0:
        return

    order = (
        torch.randperm(n_chunks_total, device=device, generator=generator)
        if shuffle
        else torch.arange(n_chunks_total, device=device)
    )

    # --- Precompute step-index grid for gather ----------------------------
    # chunk_steps[k, c] = start_t[c] + k  (shape: [chunk_total, chunks_per_batch])
    # We build this on the fly per batch.

    has_hidden = "selector_hidden" in buffer.fields

    for batch_start in range(0, n_chunks_total, chunks_per_batch):
        chunk_indices = order[batch_start : batch_start + chunks_per_batch]
        actual = int(chunk_indices.numel())
        if actual == 0:
            break

        t0 = grid_t[chunk_indices]  # (actual,)  start time-step per chunk
        e0 = grid_e[chunk_indices]  # (actual,)  environment index per chunk

        # Step indices for every position in the chunk: (chunk_total, actual)
        k = torch.arange(chunk_total, device=device).unsqueeze(1)  # (chunk_total, 1)
        step_idx = (t0.unsqueeze(0) + k).clamp(0, T - 1)            # (chunk_total, actual)

        batch: dict[str, torch.Tensor] = {}

        for field_name, field_tensor in buffer.fields.items():
            if field_name == "selector_hidden":
                continue  # handled separately below
            # field_tensor: (T, B, *extra_dims)
            extra = field_tensor.shape[2:]
            # Gather: index by time then by env
            # step_idx: (chunk_total, actual) → use to index dim 0
            # e0: (actual,) → use to index dim 1
            gathered = field_tensor[step_idx, e0.unsqueeze(0).expand(chunk_total, actual)]
            # gathered shape: (chunk_total, actual, *extra)
            batch[field_name] = gathered

        if has_hidden:
            # h_start is the hidden state at t0 for each chunk.
            # selector_hidden: (T, B, hidden_dim)
            h_field = buffer.fields["selector_hidden"]
            h_start = h_field[t0, e0]  # (actual, hidden_dim)
            batch["selector_hidden_start"] = h_start

        yield batch
