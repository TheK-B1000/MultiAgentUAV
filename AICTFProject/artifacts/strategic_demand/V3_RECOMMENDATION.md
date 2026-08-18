# V3_RECOMMENDATION — one minimal intervention

**Status: RECOMMENDATION ONLY. NOT IMPLEMENTED.** No ruleset was modified, no
PPO trained, no searcher started.

## Recommended single change

```
own_flag_home_required_to_score = True
```

**Lever 1 (own flag must be home to score). Binary. Nothing else.**

### Why this one

The V2 diagnosis shows allocation value is context-dependent but that neither
context produces BOTH legs of an opportunity cost:

```
OP6 FAST_RAID   defence pays,  offence costs nothing demonstrated
OP7 FORTRESS    offence pays,  defence buys nothing demonstrated
```

Lever 1 is the only candidate that couples the two directly. It gives
DOUBLE_BREACH a genuine downside it currently lacks — both agents forward means
a stolen home flag blocks scoring — while leaving GUARD_RAID's existing
weakness against the fortress intact. It is one boolean, so attribution stays
clean.

Verified available: the capture condition at `gpu_env/_core/_rules.py:566` is
`alive & carrying & ~tagged & (home_dist <= 1.2)` with **no own-flag check**.

### Explicitly excluded, on evidence

```
suppression_range   REMOVED — 0 suppression events / 256 episodes / all 4 ranges.
                    The M3 ladder was tuning a mechanic that never fires in 2v2.
```

This is the strongest negative result of Phase 0 and it retires a lever the
project had treated as a leading candidate.

### Held in reserve, not recommended now

- **Tag consequence duration** {1x, 2.5x, 5x} — plausible, but Q2 shows the
  tag system is already near its rate limit under BOTH_ATTACK
  (median gap ~10.4s against a 10s floor), so changing respawn may interact
  with a mechanism that is already saturated. Test after Lever 1, not with it.
- **Tag channel** {0, 1, 2}s — same reasoning.
- **Carrier vulnerability** {1.0, 0.9, 0.8} — NOT ELIGIBLE — 7E does not support the self-sufficient-carrier hypothesis, so this lever is withheld per the conditional rule.

### Why not stack them

The smallest intervention that produces strategic demand is the scientifically
interpretable one. If Lever 1 alone creates the two-way reversal, adding respawn
and channel changes would make the cause unattributable.

## What must be true before V3 is accepted

Unchanged frozen gates: two-way payoff reversal at >= 0.15 with LCB95 > 0 in
both directions, precommitment uncertainty (t_intent > t_commit) measured by a
legal-observation probe rather than a hand-picked geometry event, fresh held-out
replication, and non-degeneracy.

**Do not implement this recommendation without a human decision.**
