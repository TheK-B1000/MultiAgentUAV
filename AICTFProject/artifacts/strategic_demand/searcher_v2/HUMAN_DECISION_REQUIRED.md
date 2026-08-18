# HUMAN_DECISION_REQUIRED — first development-eligible candidate

Searcher V2 stopped immediately on the FIRST candidate to clear the frozen
Stage-2 development criteria. It did not finish the generation and it did not
compare against later candidates.

    genome     SDS2_A_payoff_INIT_3
    base       OP6
    overlay    {'min_alive_for_defender': 2}
    delta_G    +0.2188   (stage floor 0.1)
    p_C        0.9375        (stage screen 0.65)
    J_v2       +0.8828
    frac_0_0   0.359

These are DEVELOPMENT numbers, not evidence. They are not Gate B.

## Next step, if authorized

1. Freeze this genome with full lineage BEFORE any confirmation episode.
2. Re-audit the reserved block: `python scripts/audit_seed_block.py --base 6000001 --n 64`
3. Confirm at n=64 on 6000001..6000064, no extension:
   - delta_G >= 0.15 with LCB95 > 0
   - LCB95(p_C) > 0.50

No PPO, specialists, selector, FP/DO or latent training until
V3_STRATEGIC_DEMAND is validated.
