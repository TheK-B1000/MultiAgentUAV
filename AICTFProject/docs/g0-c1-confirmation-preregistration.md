# G0 → C1 confirmation preregistration

**Status:** LOCKED before any O1 training.
**Date:** 2026-07-30
**Depends on:** discovery sweep `experiments/run_g0_weakness_sweep.py`
(32 paired episodes / cell) + analyzer `experiments/analyze_g0_weakness.py`.

The running 32-seed sweep is **discovery only**. Do not train `O1` from a
discovery winner.

---

## 1. Competence (map-wide, three-way)

Count opponents with **family mean margin < 0** among the seven BASE keys
OP6–OP12 on `map_a`:

```text
0–2 negative  → COMPETENT
exactly 3     → AMBIGUOUS
4–7 negative  → INCOMPETENT
```

| Verdict | Action |
|---------|--------|
| COMPETENT | May select discovery C1 candidates that clear the weakness gate |
| AMBIGUOUS | No C1. Confirm map-wide competence with fresh seeds, or train `G0_map_a` |
| INCOMPETENT | No C1. Train a proper `map_a` incumbent (OP6–OP12 mixture) first |

Pooled map-wide mean and CI are **diagnostics only**. One easy opponent must
not hide several losses.

## 2. Discovery weakness gate (strict)

A discovery context qualifies only if:

```text
all three G0 seed means < 0
family UCB95 < 0   (strict; exact 0 fails)
```

Among qualifiers, rank mechanically:

1. lowest strongest-member margin `W(c) = max_seed payoff`
2. lowest family mean
3. lower saturation + tie rate

Behavior telemetry / trajectory fingerprints are **descriptive only** and are
not part of selection.

## 3. Confirmation block (before freezing C1)

For **every** discovery qualifier (not only the top-ranked):

```text
Policies:       all three frozen G0 members {901001, 901002, 901003}
Map:            map_a  (explicit in every row / manifest)
Opponent:       that discovery candidate
Evaluation:     64 fresh paired seeds
Horizon:        240
Deterministic:  yes; no DR; n_envs=1
```

Eval seed block (disjoint from discovery `1300001+` and all prior blocks):

```text
C1 confirmation base: 1_400_001 .. (base + 63) per candidate
```

**Confirmation gate (same strict rule):**

```text
all three G0 seed means < 0
family UCB95 < 0
```

If multiple contexts confirm, rank by the same mechanical order as discovery
(strongest-member margin → family mean → sat+tie). Freeze **one** C1.

If none confirm: do not train O1; declare a separately scoped next search
(e.g. OP6–OP12 named variants) before relaxing anything.

## 4. LRO loop after confirmation (map_a only)

```text
1. Confirm G0 competence on map_a.
2. Confirm weakness C1 independently (this document).
3. Freeze G0.
4. Train a full independent response-oracle family O1 on C1
   (multi-seed, task reward, no latent adapter / shared head).
5. Evaluate G0 and O1 on a frozen map_a acceptance pool.
6. Retain O1 only if:
     O1 reliably owns C1,
     G0 retains at least one anchor context,
     selecting between them beats either fixed family,
     and their behavior is meaningfully distinct (B_distinct-class).
7. Birth z1 only after retention passes (frozen experts; z fixed per episode).
8. Search for a weakness of the entire retained pool {G0, O1, …};
   train O2 / O3 the same way.
```

Router trains only after ≥2 retained branches. Primary latent acceptance
stays on **`map_a`**; other maps do not drive branch retention.

## 5. Prohibited

- Training O1 from the 32-seed discovery winner without confirmation
- Best-seed selection within G0
- Relaxing UCB95 ≤ 0 to “≤ 0” or manufacturing a candidate under AMBIGUOUS /
  INCOMPETENT
- Naming C1 RUSH / SPLIT / TURTLE / ESCORT before O1 behavior is observed
- Pooling map_a rows with map_b
