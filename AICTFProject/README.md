# SEA-GUARD

A maritime multi-robot capture-the-flag benchmark and evaluation protocol for
opponent-aware population training.

Autonomous surface vehicles (ASVs) play team capture-the-flag under maritime
dynamics — surface current, stochastic drift, sensor noise and dropout, tag
mechanics, and hidden mines. The contribution is **not** a new RL algorithm: PPO
is used unmodified. It is the environment, the baseline suite, the configuration
splits, and an evaluation protocol that measures three separate things instead of
one aggregate win rate.

## The evaluation triad

A single win rate against a fixed opponent cannot distinguish "competent" from
"robust". Three orthogonal questions are asked instead:

| Test | Question | Entry point |
| --- | --- | --- |
| **Performance** | How good is the policy on configurations it trained on? | `plot/eval_generalization.py` |
| **Generalization** | How much does it lose on configurations nothing trained on? | `plot/eval_generalization.py` |
| **Exploitability** | How badly can a learned best response beat it? | `plot/eval_exploitability.py` |

Exploitability is reported as an **approximate lower bound**: failing to find an
exploit at a given budget is not proof that none exists.

## Baseline ladder

Every method shares the environment, network architecture, PPO hyperparameters,
training budget, and evaluation protocol. Only the opponent-selection rule
differs, which is what makes the comparison attributable.

| Method | Opponent selection | Implementation |
| --- | --- | --- |
| Self-play | latest snapshot | `rl/train_ppo.py` |
| Curriculum league | Elo-distance matchmaking | `rl/league.py` |
| Fictitious play | uniform draw over the pool | `rl/egt_league.py` |
| Double oracle (PSRO) | meta-Nash over an empirical payoff matrix | `rl/egt_league.py` |
| PFSP | `(1 - win_rate)^p` weighting | `rl/roastar_league.py` |
| ROA-Star | PFSP + goal-conditioned exploiters | `rl/train_exploiter.py` |

## Configuration space

A configuration is drawn once per episode and held fixed:

```
c = (opponent, current profile, team size, episode seed)
```

The map and ruleset are fixed constants of the protocol, not varied factors. The
seen/held-out split is **derived from what training actually samples** rather than
chosen to hit a target size — see [`rl/configuration_space.py`](rl/configuration_space.py),
the single source of truth.

```bash
python -c "from rl.configuration_space import describe_split; print(describe_split(2))"
```

Team size is deliberately *not* a generalization axis. Observation and action
spaces are team-size dependent, so a 2v2 policy cannot be loaded at 3v3 at all;
`assert_team_size_compatible` refuses to try. Independently trained 2v2/3v3/4v4
policies are reported as a **scalability** result, not as zero-shot team-size
generalization.

## Install

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pytest tests/ plot/ -q
```

## Quickstart

```bash
# Train one baseline (2v2, fictitious play)
python rl/train_ppo_roastar.py --mode fp --agents 2 --total-steps 1000000 --seed 42

# Audit that the run is valid under its label before reporting it
python rl/verify_league_runs.py --checkpoint-root checkpoints_sb3 --strict

# Seen vs held-out configuration evaluation
python plot/eval_generalization.py --settings 2v2 --seeds 42,43,44 --episodes 200

# Approximate best-response exploitability
python plot/eval_exploitability.py --checkpoint-dir checkpoints_sb3/2v2 --list
```

Full commands, budgets, and the statistical protocol are in
[`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Validating a training run

Population-based methods can fail silently: if match results never reach the
league, PFSP weighting has no data and degenerates to a uniform draw while still
being labeled PFSP. `rl/verify_league_runs.py` audits the persisted league state
and fails any run whose sampling rule ran without the feedback it depends on. Run
it with `--strict` as a gate before any result reaches a table.

## Checkpoints

Trained checkpoints are **not** in this repository — they are tens of megabytes
each and belong in release assets, not git history. See
[`docs/REPRODUCE.md`](docs/REPRODUCE.md) for where to fetch them, or retrain from
the commands above.

## Repository layout

```
rl/                       training, leagues, configuration space, audits
  configuration_space.py  configuration definition and seen/held-out splits
  egt_league.py           fictitious play + double oracle
  league.py               Elo league
  roastar_league.py       PFSP league
  eval_exploitability.py  best-response oracle protocol
  verify_league_runs.py   league-state audit
plot/                     evaluation CLIs and figures
  eval_generalization.py  performance + generalization
  eval_exploitability.py  exploitability CLI
  eval_crossplay.py       cross-play payoff matrix
docs/paper/               formalization, related work, bibliography
tests/                    unit tests (no GPU required)
```

## Citation

Citation details will be added on publication. See `docs/paper/references.bib`
for the works this builds on.

## License

See [`LICENSE`](LICENSE).
