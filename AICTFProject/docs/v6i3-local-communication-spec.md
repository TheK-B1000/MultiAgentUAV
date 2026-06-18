# V6I3: Latent Team Strategy with Local Emergent Communication

**Status:** IMPLEMENTATION IN PROGRESS — Slices 1–2 transport/policy + Slices 3–6 phase/gates/preset landed; calibration + confirmatory run pending.

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
| `comm_num_symbols` | 4 (2 bits) |
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

(Full text as approved in the implementation brief — see repository commit / design review.)

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
