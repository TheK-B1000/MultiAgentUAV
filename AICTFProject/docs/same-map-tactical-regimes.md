# Same-map tactical regimes on `map_a` — framing lock + next experiment

**Status:** LOCKED 2026-07-30 after G0 BASE sweep COMPLETE / COMPETENT / no C1.
**Map:** `map_a` only for primary latent acceptance. Other maps do not drive
branch retention.

---

## 1. Design target

> **One map, several strategically incompatible situations, and a policy pool
> whose members solve different situations.**

Not:

```text
map_a → strategy 1
map_b → strategy 2
map_c → strategy 3
map_d → strategy 4
```

Instead:

```text
same map_a + enemy commits both attackers  → defensive / intercept response
same map_a + enemy holds deep defense      → coordinated offensive response
same map_a + teammate carries enemy flag   → escort / screen response
same map_a + own flag stolen / teammate tagged → recovery / counter response
```

These are candidate **same-map tactical regimes**. They are not declared
distinct strategies until learned policies prove complementary value.

A useful K=4 repertoire could eventually resemble:

| Situation on the same map              | Useful response             |
|----------------------------------------|-----------------------------|
| Enemy leaves home exposed              | Fast pressure / capture     |
| Enemy launches a dual rush             | Home defense / interception |
| Teammate has the flag                  | Escort / screening          |
| Enemy changes plans after commitment   | Recovery / counter-switch   |

What changes is opponent formation, flag possession, tags, tag cooldown,
score/time, teammate position, and mid-episode plan switches — not the map.

---

## 2. What the BASE G0 sweep already showed

Discovery: `artifacts/g0_weakness_sweep/` (672 eps, 3 G0 × 7 BASE × 32).

```text
competence: COMPETENT (0/7 negative family means)
no C1 under strict gate
hardest BASE: OP6 family +0.96 (still winning)
```

**Important:** short tags `OP6`..`OP12` already resolve to the seven distinct
LRO niches (`OP6_IMMEDIATE_DUAL_RUSH` … `OP12_LATE_CONVERTER`). Historical
names like `OP6_TURTLE` / `OP7_SWITCHER` are **synonyms of the same BT
profile**, not separate behaviors (`gpu_env/_core/_bt_profiles.py`).

Therefore a long-name-only re-run of the seven niches is **not** a new
scientific search for incompatible situations. It is at best a labeled
replication of the BASE niches.

**Running note (2026-07-30):** `experiments/run_g0_variant_weakness_sweep.py`
was already launched before this framing lock (`artifacts/g0_variant_weakness_sweep/`,
21 tags × 3 G0 seeds × 32 eps). Those 21 tags collapse via
`OPPONENT_SYNONYMS` / `canonicalize_opponent_key` to the **same seven BT
profiles** as BASE. Treat completion as a confirmatory labeled replication:
analyze by **canonical niche**, not as 21 independent behaviors. Do not
invent a C1 from label multiplicity. If the gate still finds nothing, the
declared successor is the **scenario bank** below — not another synonym pass
and not a new map.

---

## 3. Next experiment — same-map scenario bank (not another map)

Do **not** introduce map_b/map_c for primary acceptance. Build controlled
`map_a` situations that create commitment costs using legal game state:

```text
S1: score tied, both flags home, enemy dual rush begins
S2: BLUE ahead late, enemy launches desperate attack
S3: teammate has enemy flag, opponents converge on return path
S4: own flag stolen while both BLUE agents are forward
S5: one teammate tagged, tag cooldown active
S6: opponent opens defensively, then switches to counterattack
```

Generation (prefer legal transitions over hidden strategy labels):

* opponent opening policies;
* mid-episode opponent phase changes;
* scenario resets from valid match states;
* replaying states sampled from actual matches.

Preserve realistic commitment costs already in Aquaticus-style rules:

```text
travel time back to home flag
vehicle turning / acceleration
tag cooldown
territorial tagging
return-home after tagged
match time pressure
offensive pressure lost when one agent stays back
```

Prefer these over artificial RUSH/ESCORT reward shaping.

### Acceptance (unchanged)

For candidate O1 vs incumbent G0 on a confirmed C1:

```text
O1 reliably beats G0 on C1
G0 reliably beats O1 on ≥1 other map_a context
selecting G0 or O1 beats either fixed family
behavior meaningfully distinct (B_distinct-class)
→ retain O1 as z1; repeat for pool weaknesses → K=4
```

### Rules fidelity (diagnostic, not retuning)

Before concluding “one universal master policy,” verify the simulator
faithfully includes territory-dependent tagging, tag cooldown, return-home
after tag, safe flag return, realistic motion/turning, score and time
pressure. Fixing missing Aquaticus mechanics is fidelity, not cherry-picking.

Only after realistic same-map situations + faithful rules still yield one
master policy should we conclude the task lacks strategic tension.

---

## 4. Immediate sequence

```text
1. BASE G0 sweep on map_a          DONE — COMPETENT, no C1
2. Do NOT re-run long-name aliases as a "variant sweep"
3. Audit Aquaticus rule fidelity on map_a (diagnostic checklist)
4. Implement scenario bank S1–S6 on map_a (legal state / mid-episode switches)
5. Evaluate G0 on the scenario bank; seek C1 under the same competence + gate rules
6. Confirm C1 @ 64 fresh seeds before O1 training
7. Train full independent O1; retain only if complementary
8. Birth z0/z1; grow pool to K=4; router last
```

Launcher / analyzer for the scenario bank: TBD under
`experiments/run_g0_scenario_bank.py` (not launched until preregistered and
implemented). Until then, no O1 training and no map change.
