from dataclasses import dataclass

@dataclass(frozen=True)
class EvalCondition:
    name: str
    selection_rule: str
    strategy_interval: int
    allow_switching: bool
    description: str = ""
    fixed_latent_id: int | None = None
    latent_eval_mode: str = "normal"
    online_rollout: bool = True
    identity_assisted: bool = False
    posthoc_only: bool = False

def validate_condition(condition: EvalCondition) -> None:
    if condition.allow_switching and condition.strategy_interval <= 0:
        raise ValueError(
            f"{condition.name}: switching requires strategy_interval > 0"
        )
    if not condition.allow_switching and condition.strategy_interval != 0:
        raise ValueError(
            f"{condition.name}: non-switching condition must use interval 0"
        )
