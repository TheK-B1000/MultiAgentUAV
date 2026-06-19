# V6I3: Latent Team Strategy with Local Emergent Communication

**Status:** IMPLEMENTED WITH FROZEN CONFIRMATORY CONTRACT — Slices 1–6
and the `v6i3` preset are landed. The confirmatory contract below is
frozen as of 2026-06-18; a full fresh confirmatory run is still pending.
Any v6i3 artifacts produced before this freeze are exploratory only.

**Protocol ID:** `v6i3_strategy_local_comm_v1`

**Parent:** v6i2 dual-evidence staged curriculum (`gate_protocol_version = v6i2_dual_evidence`).

**Invariant:** v6i1 and v6i2 lineages remain unchanged when `communication_enabled=False`.

This document owns the scientific and implementation contract for V6I3. See also
[`v6i2-gate-protocol-freeze.md`](v6i2-gate-protocol-freeze.md) for the strategy-only baseline.

---

## 1. Research objective

V6I3 extends the v6i2 latent-strategy system with a constrained, learned agent-to-agent
communication channel.

The system should learn:

1. A shared team-level strategy: \(z \in \{0,1,2,3\}\)
2. A local message emitted by each agent: \(m_i \in \{0,1,2,3\}\)
3. A decentralized action policy: \(\pi_\theta(a_i \mid o_i, z, M_i)\)

where \(M_i\) contains messages received from nearby teammates.

Neither \(z\) nor \(m_i\) receives a handcrafted semantic label.

**Execution claim:** centralized low-bandwidth strategy selection with decentralized
local communication and decentralized action execution.

---

## 2. Locked communication channel

| Parameter | Value |
|-----------|------:|
| `comm_num_symbols` | 5 (`SILENCE` + 4 two-bit active symbols) |
| `comm_silence_symbol` | 0 |
| `comm_interval_steps` | 32 |
| `comm_delivery_delay_steps` | 1 |
| `comm_radius_cells` | 6.0 |
| `comm_dropout_probability` | 0.10 (training only) |
| `comm_entropy_coef` | 0.001 |

Messages persist until the next communication boundary. Delivery at \(t+1\) for sends at \(t\).
Recipient set fixed at send time. Dropout is per sender–receiver pair. No fifth symbol for drop.

**PPO contract (boundary-only credit):** a held outbound symbol persists for
`comm_interval_steps` decision steps in the transport/observation path, but the
message head receives exactly **one** policy-gradient draw per hold window — on
the send boundary row only. Non-boundary rollout rows store
`message_log_probs = 0`, `message_boundary_mask = false`, and PPO replay skips
the message-head forward pass for those rows.

---

## 3. Message representation

Four additional local spatial CNN channels (one per symbol). Sender position marked in the
receiver's egocentric grid for the appropriate symbol channel. No hidden-state broadcast.

---

## 4–17. Policy, PPO, CF boundary, curriculum, telemetry, evidence, ablations, config, checkpoints, tests, launch

### 4. Frozen preset identity and lineage

| Property | Value |
|----------|-------|
| Apply function | `apply_plan_faithful_latent_v6i3_strategy_local_comm` |
| Primary preset | `v6i3_strategy_local_comm` |
| Aliases | `plan_faithful_latent_v6i3_strategy_local_comm`, `latent_v6i3_strategy_local_comm`, `v6i3_strategy_local_comm`, `v6i3_local_comm`, `v6i3` |
| Parent | `apply_plan_faithful_latent_v6i2_staged_team_intent_curriculum` |
| Classification | `SUMMER-COMPATIBLE EXTENSION` |
| Confirmatory role | Official v6i3 local-communication row only for runs launched after this freeze |
| Parent gate fingerprint | `85506ab324d464c5` |
| V6I3 gate fingerprint | `f458d26cd040232d` |
| Run tag | `v6i3_strategy_local_comm_OP5_OP6_OP7_1m_4v4` |

The resolved diff against v6i2 is exactly the communication protocol and
artifact identity surface: `experiment_id`, `gate_protocol_version`,
`communication_enabled`, `comm_protocol_version`, the communication
channel fields in section 2, the frozen communication gate fields in
section 5, and `run_tag`. v6i3 inherits v6i2's strong-CF ceiling,
dual-evidence strategy gates, A/B/C schedule, opponent pool, split-lane-v2
map, latent contract, and confirmatory resume safety.

### 5. Frozen communication evidence gates

These values are part of the resolved v6i3 gate fingerprint. Changing
any value after inspecting confirmatory results invalidates the
confirmatory lineage and requires a new protocol/fingerprint.

| Gate field | Locked value | Meaning |
|------------|-------------:|---------|
| `comm_min_valid_boundaries` | 1024 | Minimum communication send-boundary rows in the rollout gate sample |
| `comm_min_deliveries` | 4096 | Minimum delivered sender-receiver messages |
| `comm_min_symbols_used` | 2 | At least two active non-silence symbols must appear |
| `comm_entropy_floor` | 0.0 | Entropy is diagnostic during Phase A |
| `comm_symbol_dominance_ceiling` | 1.0 | Dominance is diagnostic during Phase A |
| `comm_listener_jsd_margin` | 0.001 | Diagnostic mean listener action-distribution JSD floor under message-symbol intervention |
| `comm_listener_min_states` | 64 | Minimum listener intervention states sampled for the diagnostic |
| `comm_listener_min_passing_pairs` | 3 | At least 3 of 6 symbol pairs must clear the JSD margin |
| `comm_listener_consecutive_updates` | 1 | Diagnostic listener-intervention batch count |

The Phase A usage gate must pass activity, delivery, and active-symbol
coverage checks. Listener causal response, entropy, dominance, corruption
loss, and semantic message-role associations are logged diagnostics during
Phase A; they do not block promotion. Final V6I3 communication-value claims
require listener response, matched-seed silence/shuffle degradation, and
value over the v6i2 strategy-only baseline.

### 6. Corruption reliance gate

Corruption tests are post-training confirmatory evidence, not a Phase-A
promotion signal. A v6i3 result may be called communication-dependent only
if a matched-seed evaluation with at least 100 episodes per mode shows a
performance loss under message-channel corruption.

Frozen threshold:

```text
natural - corrupted mean episode return >= 0.02
```

The threshold must hold for at least two corruption modes from
`silence`, `shuffle`, `random`, `constant`, and `extra_delay`, and the
paired bootstrap 95 percent confidence interval on the natural-minus-
corrupted delta must exclude zero. If the report uses win rate instead
of return, the fallback threshold is an absolute WR drop of at least
0.03 with the same paired-bootstrap requirement.

### 7. Confirmatory launch contract

The official v6i3 run must be a fresh run after this freeze:

```bash
python rl/train_ppo.py \
  --preset v6i3 \
  --total-steps 1000000 \
  --agents 4 \
  --seed 0 \
  --device cuda \
  --n-envs 32 \
  --checkpoint-dir checkpoints/4v4 \
  --fresh-metrics-csv \
  --periodic-checkpoint-steps 50000
```

The run is confirmatory only if `confirmatory_gate_lineage_valid=True`,
`allow_gate_config_mismatch_on_resume=False`, the recorded
`gate_config_fingerprint` is `f458d26cd040232d`, and the v6i2 parent
fingerprint is still `85506ab324d464c5`. Resuming with a gate-config
mismatch override makes the run exploratory.

### 8. Implementation slices

Implementation slices:

1. **Transport** — `rl/custom_ppo/communication/` (this slice)
2. Policy and rollout plumbing
3. Phase integration
4. Telemetry
5. Corruption evaluator
6. V6I3 evidence protocol

---

## Changelog

| Date | Notes |
|------|-------|
| 2026-06-18 | Spec document created; Slice 1 transport module + unit tests |
| 2026-06-18 | Slice 2: message head, rollout buffer fields, boundary-masked PPO, checkpoint comm state |
| 2026-06-18 | Slices 3–6: phase freeze, telemetry/MI/listener, corruption runtime, v6i3 gate protocol + preset |
| 2026-06-18 | Frozen confirmatory lineage: v6i3 parent/fingerprint, nontrivial communication gates, and corruption reliance threshold |
| 2026-06-18 | Phase A gate revised to require communication transport but keep listener causal value as diagnostic; vocabulary changed to `SILENCE` + four active symbols |
