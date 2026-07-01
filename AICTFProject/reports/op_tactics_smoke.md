# OP5..OP12 tactical evaluation

| Opponent | Escort | Intercept | Counter | ObjChg | Stuck | Route div | ATT% | DEF% | ESC% | INT% | CTR% |
|----------|--------|-----------|---------|------|-------|-----------|------|------|------|------|------|
| OP8 | 0.0 | 0.0 | 0.0 | 3.0 | 1.0 | 1.44 | 79 | 0 | 0 | 0 | 0 |
| OP9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.79 | 100 | 0 | 0 | 0 | 0 |
| OP10 | 33.0 | 0.0 | 0.0 | 2.0 | 1.0 | 1.55 | 67 | 0 | 33 | 0 | 0 |
| OP12 | 0.0 | 0.0 | 0.0 | 5.0 | 0.0 | 1.45 | 69 | 0 | 0 | 0 | 0 |

_Role percentages are time-averaged agent-role occupancy during passive-blue rollouts. Escort/intercept/counter columns are cumulative BT telemetry counters._

# Contested scenario battery (forced geometries)

| Opponent | Intercept | Counter | Escort | Dual-carrier | OP9 mine |
|----------|-----------|---------|--------|--------------|----------|
| OP8 | 1 | 0 | 0 | 0 | 0 |
| OP9 | 1 | 0 | 0 | 0 | 1 |
| OP10 | 1 | 0 | 1 | 1 | 0 |
| OP12 | 1 | 1 | 1 | 1 | 0 |

_Each cell is 1 if the scenario fired the expected role/signal for that opponent, else 0. ``counter_infeasible`` expects COUNTER without INTERCEPTOR (OP12-style geometry). ``op9_mine_intent`` is only meaningful for OP9 (others may read 0)._