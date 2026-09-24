# Scaling audit: does the policy interface still express the strategy at 4v4/6v6?

**Date:** 2026-09-12 · **Type:** static code audit, no GPU, no seeds spent
**Status:** findings are STRUCTURAL. They establish what the interface *can* carry,
not what the trained policies actually did. Nothing here is a measured result.

## The distinction this audit exists to serve

    strategic demand  ≠  strategy representability  ≠  strategy learnability

- **Demand** — certified at 4v4 and 6v6 by the scripted-probe reversal. A
  guard-style scripted response beats a breach-style one against pole A and
  vice versa against pole B. **Established.**
- **Representability** — can *our* observation/action interface, with *our*
  weight-shared actor, express the 4v4/6v6 version of that strategy at all?
  **This audit addresses this, statically. It is the gap that was being skipped.**
- **Learnability** — can PPO acquire it. Everything we have been diagnosing
  (SNR, trajectory, ablation) lives here, and it is only the right question
  *after* representability is established.

The scripted certification proves demand exists between **two hand-coded
controllers**. It says nothing about whether a shared-weight neural policy
reading our observation can represent either of them at N=6.

---

## Finding 1 — the identity channel does not scale with team size

`gpu_env/_core/_observations.py:195-196`

```python
agent_id = torch.arange(n_agents, device=self.device, dtype=torch.float32)
out[..., 13] = agent_id[None, :] / max(1.0, float(n_agents - 1))
```

Symmetry-breaking machinery **does** exist — one scalar, vec feature 13. But it
is **one dimension regardless of N**, and the identities it must separate are
collinear points on that single axis:

| Team size | agent_id values | spacing | roles to distinguish |
|---|---|---|---|
| 2v2 | 0.0, 1.0 | **1.00** | 2 |
| 4v4 | 0.0, 0.33, 0.67, 1.0 | 0.33 | 4 |
| 6v6 | 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 | **0.20** | 6 |

At 2v2 this is effectively a dedicated binary flag — maximally separated, trivially
usable by a shared network to run two different behaviours. At 6v6 the same
network must partition one scalar into six bands *and* bind each band to a
different role, with 5× less margin. Identity **channel capacity is constant
while the required identity resolution grows with N.**

A one-hot agent id (N dims) or a learned per-agent embedding would give
orthogonal identities. A normalized scalar gives collinear ones.

## Finding 2 — teammates are an anonymous, stateless position blob

`gpu_env/_core/_observations.py:82-84`

```python
friend_live = own_alive.clone()
friend_live[:, i] = False
self._scatter_points(grid[:, i], 1, own_x_obs, own_y, friend_live)
```

Channel 1 is an **unordered set-scatter of teammate positions with self
excluded** — structurally identical to the enemy channel 2. And all 20 vec
features are self-referential (own pose, own payload, own mine charges, own
score) plus nearest-*enemy* distance (11) and nearest-*friendly-mine* distance
(17).

So the total allocation-relevant information an agent has about its teammates is:

| Information | Available? |
|---|---|
| teammate positions | ✅ but **unordered, anonymous** |
| which teammate is which | ❌ no identity anywhere |
| teammate payload / carrying state | ❌ |
| teammate committed macro / intent | ❌ |
| nearest-teammate distance | ❌ (exists for enemies and mines, not teammates) |

An agent therefore **cannot** reason *"a teammate is already going for the flag,
so I should guard"* — not because it is hard, but because the required facts are
absent from the observation.

**The 2v2 special case is what makes this decisive.** At 2v2 that unordered
teammate set has **exactly one element**, so anonymity costs nothing: "the blob"
*is* my only teammate, unambiguously. Paired with a clean binary agent_id, a 2v2
agent has effectively complete team state. At 6v6 the identical code yields an
anonymous crowd of five. **The interface does not degrade gracefully with N; it
changes kind.**

## Finding 3 — the reward is a single team scalar, with 1/N shaping dilution

`gpu_env/_core/_step.py:483`

```python
rterm = torch.zeros((self.B,), dtype=torch.float32, device=self.device)
```

Shape is `(B,)` — **one reward per environment, not per agent**. Every agent
receives the identical scalar. Additionally, the shaping terms in
`gpu_env/_core/_rewards.py` collapse the agent dimension with `.mean(dim=1)`
(lines 54, 63, 76, 94, 178, 207, 216):

```python
attack_phi = torch.where(blue_carrying, zeros, closeness(attack_dist)).mean(dim=1)
```

So one agent's contribution to any shaped term is **1/N of a team average** —
1/2 at 2v2, 1/6 at 6v6, a 3× weaker per-agent signal. Terminal win/loss
(`win_team_reward`) is shared undiluted, which is correct for a team game but
provides no role-differentiating information.

Combined with weight sharing, this means: an agent that correctly specializes
receives a signal in which ~1/6 is its own contribution and ~5/6 is teammate
behaviour it neither controls nor observes. **This is a concrete,
code-grounded version of the credit-assignment story — stronger than the cited
gradient-variance theorem, because it is our actual reward function.**

## Finding 4 — N-invariant commitment horizons mean worse coordination latency

`gpu_env/_config.py:190-194` — commit horizons are fixed constants
(GO_TO 4, GRAB_MINE 3, GET_FLAG 4, PLACE_MINE 2, GO_HOME 4 ticks) **regardless
of N**. Between one agent's decisions, the number of other agents that have moved
scales with N: 3 others at 2v2, 11 others at 6v6. The world an agent committed
against is proportionally staler at larger N.

## Finding 5 — fixed 20×20 arena, tripled occupancy

`gpu_env/_constants.py:3-4` — `CNN_COLS = CNN_ROWS = 20`, fixed. Total agents go
4 → 8 → 12 on the same grid at the same resolution. Two consequences: physical
congestion, and **scatter collisions** — multiple teammates landing in the same
cell of channel 1 become more likely, further degrading an already anonymous
representation.

## Finding 6 — nothing in the interface prevents duplicated objectives

There is no mutual-exclusion mechanism, no claim/lock on a target, and no
teammate-intent channel. Two agents selecting `GET_FLAG` with the same target
is fully representable and invisible to each of them at selection time. Whether
this *happens* requires behavioural data (see open questions).

---

## What degrades from 2v2 to 6v6, summarized

| Mechanism | 2v2 | 6v6 | Scales? |
|---|---|---|---|
| agent identity resolution | binary, spacing 1.0 | 6 levels, spacing 0.2 | ❌ 1/(N−1) |
| teammate set ambiguity | 1 element, unambiguous | 5 anonymous | ❌ qualitative |
| teammate state/intent | absent (but only 1 teammate) | absent (5 teammates) | ❌ |
| per-agent share of shaped reward | 1/2 | 1/6 | ❌ 1/N |
| per-agent reward differentiation | none (team scalar) | none (team scalar) | ❌ constant zero |
| commitment staleness | 3 others move | 11 others move | ❌ |
| arena occupancy | 4 on 20×20 | 12 on 20×20 | ❌ |
| actor parameters | 3,473,592 | 3,473,592 | — invariant by design |

Every row that could support role differentiation gets worse with N, and the one
row that is invariant is the parameter count.

---

## What this does NOT establish

- That the trained 4v4/6v6 policies actually failed to differentiate *for these
  reasons*. This is a static audit of available mechanisms, not a measurement.
- That the interface is *insufficient*. A thin mechanism is not a proven
  impossible one — a network could in principle resolve six bands of one scalar.
  Sufficiency is an empirical question, which is exactly what the behaviour-
  cloning diagnostic below settles.
- Anything that reinterprets a sealed result.

---

## The diagnostic this points to: behaviour cloning the scripted strategy

Take the scripted GUARD/BREACH controllers that **already demonstrated demand**
at 4v4/6v6 and ask whether our exact neural policy can reproduce them with
learning difficulty removed:

1. Roll out the scripted controllers, recording `(o_i, z) → a_i` using the
   **same observation builder and the same legal macro space** PPO sees.
2. Train **pure supervised behaviour cloning**. No reward, no PPO, no credit
   assignment, no exploration.
3. Evaluate **closed-loop** against the poles.

The three outcomes separate cleanly, and each licenses a different next step:

| Outcome | Diagnosis | Implication |
|---|---|---|
| Cannot fit offline | **representation/interface problem** | observation, identity, or action space is inadequate — add per-agent role code / teammate state / richer opponent structure |
| Fits offline, fails closed-loop | **state-distribution / role-allocation problem** | compounding error and coordination, not expressiveness — allocation mechanism needed |
| Fits and executes closed-loop | **interface is adequate** | representability established ⇒ PPO acquisition becomes the prime suspect, and *only then* does gradient-SNR deserve compute |

This is strictly more informative than measuring gradient SNR first, because in
the first two cases an SNR measurement would be **diagnosing the optimizer for a
failure that is not the optimizer's** — and, worse, a low-SNR reading in those
cases would look like confirmation.

### Why this subsumes the SNR question rather than competing with it

`GRAD_SNR_SCALING` is frozen, validated (115 tests) and **not launched**. Its own
gate already requires PI authorization. This audit adds a reason to keep it
gated: its entire hypothesis space presumes the strategy is representable. The BC
diagnostic tests that presumption for far less compute than a training
intervention, and its third outcome is precisely the precondition SNR needs.

**Revised priority:** BC representability probe → (if adequate) SNR / acquisition
→ (if inadequate) interface changes. SNR moves down one position, as directed.

---

## Open questions requiring behavioural data, not code reading

1. Do multiple agents in fact select identical objectives simultaneously, and how
   often, at 4v4 vs 6v6? (Measurable from existing `episode_rows.csv` /
   rollout traces without new training.)
2. Is the six-level `agent_id` scalar actually *used* by the trained policies —
   i.e. does perturbing feature 13 change the action distribution? (A cheap
   ablation in the same family as the running opponent-channel ablation.)
3. Do the poles demand different *learnable* Blue responses, or only different
   hand-coded ones? This is the deepest question and is exactly what BC answers.
4. What do GUARD and BREACH mean at N=4 and N=6 — how many agents per role, and
   are roles fixed or dynamic? Requires reading the scripted controller
   definitions, which is the next audit step.

---

## Provenance

Verified by reading: `gpu_env/_core/_observations.py`,
`gpu_env/_core/_rewards.py`, `gpu_env/_core/_step.py`,
`gpu_env/_constants.py`, `gpu_env/_config.py`. Line numbers are as of git
`school-testing` on 2026-09-12. No seeds spent; the live
`OPP_ABLATION_6V6` run was not touched.
