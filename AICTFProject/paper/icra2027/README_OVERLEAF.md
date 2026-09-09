# ICRA 2027 Overleaf package (source of truth)

**Path:** `AICTFProject/paper/icra2027/`  
**Zip:** `AICTFProject/paper/icra2027_overleaf.zip`  
**Experimental framing:** FROZEN (`paper/data/icra_spine_freeze.json`)

## Create the Overleaf project

1. Go to [overleaf.com](https://www.overleaf.com) → **New Project** → **Upload Project**.
2. Upload `icra2027_overleaf.zip` (or zip this `icra2027/` folder yourself).
3. Set the main document to `main.tex`.
4. Invite collaborators; treat this Overleaf project as the shared editing surface.
5. Keep repo copies under `paper/icra2027/` in sync when you pull edits back.

We cannot create your Overleaf account project from this machine without your login; upload is the supported path.

## Document order (compile order in `main.tex`)

| Order | File | Notes |
|------:|------|-------|
| 1 | `sections/01_abstract.tex` | Final polish last |
| 2 | `sections/02_introduction.tex` | Keep research question |
| 3 | `sections/04_related_work.tex` | Stub — expand citations |
| 4 | `sections/03_problem_setup.tex` | Reproduce-ready definitions |
| 5 | `sections/05_methodology.tex` | Recipe, no experiment archaeology |
| 6 | `sections/06_results.tex` | **Terminology block at top**; figures speak robot |
| 7 | `sections/07_discussion.tex` | |
| 8 | `sections/08_conclusion.tex` | |

**Writing polish order (as agreed):** Problem Setup → Methodology → Results → Discussion → Conclusion → Related Work / Abstract.

## Reviewer-facing terminology (Results)

Defined at the opening of Results so plots never require repo slang:

- **Specialization / crossover** — both \(\Delta_A,\Delta_B\) positive with \(\mathrm{LCB}_{95}>0\)
- **Satisfies / fails criterion** (figure labels pass/fail) — that pre-specified test, not vibes
- **Detectable loss** — \(\mathrm{UCB}_{95}(D)<0\) vs Share-0
- **Contested / unsaturated** — not floor/ceiling win rates
- **Certified** — demand protocol only; not learned specialization

## Figures included

`figures/*.pdf` — pipeline, Claim A, absolute win-rate context (`fig_absolute_winrate_context_2v2`), sharing ladder, qualitative, trajectory strip, robustness, measurement hierarchy, Claim B ablation.

## Sync rule

- **Editing for ICRA:** prefer Overleaf / `paper/icra2027/`.
- Legacy monolith `paper/methodology.tex` was synced once for Results terminology; prefer the split package going forward.
