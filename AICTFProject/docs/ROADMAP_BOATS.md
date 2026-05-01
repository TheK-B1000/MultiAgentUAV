# Roadmap: sim foundation → cooperative boat agents

This file is a **commitment on disk**, not a forecast. It survives context loss, new chats, and six months of distraction. When a change looks tempting (critic tuning, fancier latent heads, leaderboard chasing), it should pass the rule below first.

## Gap filter (read before each experiment)

**Does this close a sim-to-real gap, or does it only make the sim number prettier?**

If it is only the latter, defer it until boat-relevant debts (robustness, dynamics diversity, controller realism, field integration) are in better shape.

---

## Why boats are a plausible next target

Compared with aggressive aerial control, surface vessels are often **friendlier for transfer**: slower motion (errors accrue more gently), mature hydrodynamics models, GPS and comms that usually work on water. Hard parts include **wind/current**, **partial observability of other craft** (sensor noise and dropout in training already touch this), and the **actuator gap**: policy commands in sim vs rudder/throttle (or twin-screw) commands on hardware, including latency, saturation, and tracking error. The **controller layer**—and later, **randomizing or modeling command-following error** after empirical characterization—is the highest-leverage prep beyond policy DR alone. If a real or RC platform exists, **measure its step response** before baking assumptions into sim; then make execution stochastic in ways that match measurement.

**Logged real trajectories** (when available): use for sim fidelity checks (qualitative match under similar conditions) and, at a more advanced stage, residual dynamics correction. Not a near-term build; worth remembering.

---

## Phase plan (roughly 6–12 months)

Each phase should produce a **named artifact** (plots, tables, checkpoint + report, controller spec, field log) so work is reviewable and shippable—not one undifferentiated push.

### Months 1–2: Close methodology debt on the current sim

**Goal:** Robustness and baselines on the existing CTF sim.

**Includes:** DR robustness curves (WR vs perturbation level); opponent pool or broader scripted variation; no-latent baselines under DR. See `docs/METHODOLOGY_DOMAIN_RANDOMIZATION.md` and `docs/SETUP_AND_TRAIN.md`.

**Artifacts:** Curves and CSVs; short interpretation notes; 200k DR sanity before committing to 1M DR.

### Months 3–4: Sim-to-sim transfer

**Goal:** Show that robustness **generalizes to different dynamics**, not only noisier same-dynamics.

**Includes:** Train in one sim (or profile); evaluate in a second (e.g. perturbed marine-style dynamics or another toolkit). This is a **credibility gate** for transfer claims.

**Artifacts:** Side-by-side WR / return comparison; clear statement of what differed between sims.

### Months 5–6: Controller layer design and characterization

**Goal:** Bridge abstract policy outputs to low-level actuation.

**Includes:** With hardware access—measure response, replicate statistics in sim, retrain. Without hardware—define the abstraction and randomize over **plausible** tracking-error parameters until data exists.

**Artifacts:** Controller interface spec; parameter ranges tied to measurement or explicit assumptions; updated sim hooks if needed.

### Months 7+: Real platform integration

**Goal:** Deliberately easy scenarios first, with safety culture.

**Includes:** Controlled water, manual override, geofencing, logging; expand difficulty only after stable behavior.

**Artifacts:** Field logs, incident checklist, versioned policy + sim commit used for each trial.

---

## Relationship to the CTF project today

The **original** arc was latent-strategy MARL in a CTF sim—that foundation is in place. The **current** arc is using that foundation for **cooperative agents that can transfer** toward real boats. The durable asset is the **apparatus**: diagnostics, DR protocol, evaluation discipline, information-ceiling awareness—not any single win-rate number.

**Training commands** remain in `docs/SETUP_AND_TRAIN.md`. **DR interpretation** lives in `docs/METHODOLOGY_DOMAIN_RANDOMIZATION.md`.
