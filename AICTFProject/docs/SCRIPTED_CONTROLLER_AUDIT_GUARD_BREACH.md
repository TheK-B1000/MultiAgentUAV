# Controller-to-observation audit: what GUARD and BREACH actually require

**Date:** 2026-09-12 · **Type:** static code audit, no GPU, no seeds
**Purpose:** determine what information the scripted teachers use, before writing
any behaviour-cloning objective. Designed around the code that implements
GUARD/BREACH, not around our conceptual idea of them.

## Where the controllers live, and what they emit

- `GUARD = "BLUE_ONE_DEFENDER_V2"`, `BREACH = "BLUE_BOTH_ATTACK_V2"`
  (`experiments/strategic_demand_searcher.py:33-34`)
- Implemented **inside the env core**: `gpu_env/_core/_scripted_blue_styles.py`,
  driven by `core.blue_scripted = True` / `core.set_blue_style(style)`.
- The certification steps the env with **zeroed actions**
  (`env.action_space.sample() * 0`, `strategic_demand_searcher.py:84`) — the
  macro action space is bypassed entirely.
- Each style is *"a hand-coded target-selection policy (per-agent tx, ty)"* —
  they emit **continuous target coordinates**, not macros.

---

## THE HEADLINE: the multi-defender assignment machinery is dead code at N=2

`_blue_one_defender_v2_targets`, lines 169-206:

```python
n_def = (N + 1) // 2
lo = N - n_def
...
def_px = own_x[:, lo:]                      # defender subset, BY INDEX
taken  = torch.zeros((B, n_def), dtype=torch.bool)
n_slots = min(n_def, int(order.shape[1]))
for k in range(n_slots):
    ...
    dd = self._dist(def_px, def_py, tx[:, None], ty[:, None])
    dd = torch.where(taken, torch.full_like(dd, big), dd)   # exclude assigned
    pick = dd.argmin(dim=1)                                  # nearest UNASSIGNED defender
    ...
    taken = taken | sel
```

Trace `n_def` and the loop by team size:

| N | n_def | defender indices | loop iterations | `taken` mask | `dd.argmin` over |
|---|---|---|---|---|---|
| **2** | 1 | `[1]` | **1** | 1 slot — **never excludes anything** | **1 element — always 0** |
| **4** | 2 | `[2, 3]` | 2 | **live: iteration 2 excludes iteration 1's pick** | 2 elements |
| **6** | 3 | `[3, 4, 5]` | 3 | live, 3-way | 3 elements |

**At N=2 the entire allocation mechanism is vacuous.** One defender, one loop
pass, an `argmin` over a single element, and a `taken` mask that can never
exclude anybody. The code's own comment confirms this is deliberate:

> *"At N=2 there is one defender and one threat slot, so the sole defender takes
> the threat nearest home -- exactly V1. The N=2 path is bit-identical, not
> merely similar."*

**At N=4 it switches on for the first time.** This is the answer to "what changed
between 2v2 and 4v4": the teacher stopped being a per-agent reactive rule and
became a **centralized sequential assignment algorithm**.

The codebase already measured the consequence independently — the comment at
line 157-158 cites `DEFENDER_STACKING_DIAGNOSTIC.json`:

> *"multi-threat contested steps are 9.55% at 2v2 but 31.08% at 4v4"*

A 3.3× increase in genuinely contested multi-threat situations from 2v2 to 4v4.

---

## Teacher decision inputs vs. student observation

### GUARD (`BLUE_ONE_DEFENDER_V2`)

| Teacher decision input | In student obs? | Quality at N≥4 |
|---|---|---|
| own position (exact float) | ✅ vec 0-1 | good |
| own role = "is my index ≥ lo" | ✅ vec 13 (agent_id) | weak scalar, spacing 0.33 (N=4) / 0.20 (N=6) |
| own flag home | ✅ grid ch 5, vec 6-7 | good |
| enemy flag position | ✅ grid ch 6, vec 4-5 | good |
| own carrying (carrier-evasion priority) | ✅ vec 10 | good |
| **which teammates are DEFENDERS** (`own_x[:, lo:]`) | ❌ | **grid ch 1 mixes defenders + attackers into one anonymous set** |
| **exact positions of the other defenders** (`def_px`) | ❌ | cell-quantized, unlabeled |
| **which defenders are already assigned** (`taken`) | ❌ | **pure intra-step centralized state — not in any observation, even in principle** |
| **a SPECIFIC enemy selected by index** (`enemy_x.gather(1, tidx)`) | ❌ | anonymous scatter; no per-enemy identity |
| **per-enemy tagged status** (`enemy_tagged`) | ❌ | absent as a distinct signal |
| threat ranking by distance to our home | ~ | inferable from ch 2 + ch 5, at grid resolution only |

Classification:
- **fully observable:** own pose, own role bit, flag positions, own carrying
- **weakly observable / inferable:** threat ranking (grid-quantized)
- **privileged / unobservable:** defender-subset membership, other defenders'
  exact positions, the `taken` assignment mask, per-enemy identity and tagged
  status

### BREACH (`BLUE_BOTH_ATTACK_V2`)

```python
def _blue_both_attack_v2_targets(self, enemy_flag_pos, B, N):
    """Both vehicles pursue the enemy flag. No defensive commitment."""
    target_x = enemy_flag_pos[:, 0:1].expand(B, N).clone()
    target_y = enemy_flag_pos[:, 1:2].expand(B, N).clone()
```

**Fully observable at every N.** A constant function of the enemy flag position,
which is in both the grid (ch 6) and the vec (4-5). No teammate information, no
enemy information, no index, no state. Trivially representable and trivially
clonable.

---

## The consequence that matters

The strategic contrast that creates measured demand is
`GUARD − BREACH`. BREACH is trivial at all N. Therefore:

> **At N≥4, the entire strategic distinction the poles reward is carried by the
> component of the teacher that the student cannot observe.**

At N=2 this was not true: with one defender, GUARD's rule reduces to *"go to the
threat nearest home, clamped to a disc"* — computable from the enemy scatter and
own flag home, requiring **no teammate information at all**. Both roles were
fully representable from the student's observation. The interface was adequate
because the coordination problem was absent.

## Second, independent gap: the action spaces do not match

| | Teacher | Student |
|---|---|---|
| output | continuous `(tx, ty)`, moving with the threat | `MultiDiscrete([n_macros, n_targets])`, `n_targets = 50` "fixed positions sampled over the map" (`gpu_env/_config.py:177-178`) |

GUARD's defender target is `home + (threat − home) * clamp(radius/dist, max=1)` —
a continuous point on a disc around home that tracks a moving intruder. The
student can only emit one of **50 fixed map waypoints**. Even with perfect
information, the student may be unable to *express* the teacher's target;
projecting teacher labels onto `(macro, target_idx)` is lossy and possibly
non-injective.

---

## What this means for the BC probe design — it must change

The PI's warned-of failure mode is **real and now localized**. If we had trained
BC on `o → a` we would have been fitting an ill-posed target: the same student
observation is consistent with different teacher actions, because the teacher's
choice depends on defender-subset membership and the `taken` mask, neither of
which is observable. The loss would have plateaued at an irreducible floor and we
would have misread that floor as *network capacity*.

**Revised design — three arms, in this order:**

1. **Label-consistency probe first (cheapest, no training).** Roll out the
   teacher, record `(student_obs, teacher_target)` pairs, and measure how often
   near-identical student observations carry materially different teacher
   targets. This *measures* the non-functionality rather than assuming it. If the
   collision rate is high, BC's loss floor is explained before we train anything.
2. **Privileged-teacher BC (representability upper bound).** Clone with the
   teacher's *own* inputs exposed (defender membership, labeled teammate
   positions, per-enemy identity). If this fits and executes, the strategy is
   representable **given adequate information**, isolating the defect to the
   observation interface rather than the network or the action space.
3. **Standard-observation BC (the actual interface test).** Clone with only what
   PPO sees. The gap between arms 2 and 3 is a direct, quantitative measure of
   **how much of the strategy our current interface destroys.**

Arm 3 alone would have been uninterpretable. Arms 2−3 together are the
experiment.

---

## Priority: 4v4 is the right laboratory

The scaling boundary is now localized to **N=2 → N=4**, not to 6v6:

| | assignment machinery | strategy representable from student obs? | empirical |
|---|---|---|---|
| 2v2 | vacuous (1 defender) | **yes** — no teammate info needed | specialization works |
| 4v4 | **activates** (2 defenders, `taken` live) | **no** — needs defender subset + assignment state | weak / not established |
| 6v6 | 3-way | no | not established |

4v4 is the first failing scale, is cheaper, and is one step from the working
baseline. Fix there, then test whether the fix ports to 6v6 and moves the
boundary.

### Minimal-intervention candidates this audit implies

Ordered by how directly they close a gap the teacher actually needs, each
testable against the 2v2-working baseline:

1. **Role/assignment code in the observation** — give each agent its assigned
   role (and optionally its assigned threat slot) as an explicit input. This is
   the smallest change that closes the largest gap, and it mirrors what the
   teacher has. Not "cheating": the teacher has it, and the 2v2 case got it for
   free from a 1-bit identity.
2. **Labeled/structured teammate representation** — per-teammate entries
   (position, role, payload) instead of one anonymous occupancy channel. Closes
   the defender-subset gap.
3. **One-hot or embedded agent identity** instead of a collinear scalar — closes
   the identity-resolution decay independently of the rest.
4. **Per-agent or role-attributed reward** instead of a single team scalar with
   `.mean(dim=1)` shaping — closes the 1/N credit dilution.

Each is a candidate, none is authorized. The BC arms above determine which gap
actually binds before any training is spent.

---

## Provenance

`gpu_env/_core/_scripted_blue_styles.py` (styles, dispatch, V2 implementations),
`experiments/strategic_demand_searcher.py` (GUARD/BREACH constants, run_episode),
`experiments/certify_strategic_demand_scaled.py` (defender-count rule),
`gpu_env/_config.py` (n_targets, commit horizons), `gpu_env/_envs.py` /
`gpu_env/_specs.py` (action space). Git `school-testing`, 2026-09-12. No seeds
spent; the live `OPP_ABLATION_6V6` run was not touched.
