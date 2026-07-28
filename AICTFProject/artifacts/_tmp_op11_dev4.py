import pandas as pd

df = pd.read_csv("artifacts/op11_dev4_predictive_dual_deny_8seed/episode_results.csv")
print("n=", len(df))
print(df.groupby("blue_style")[["win_margin", "success", "blue_score", "red_score"]].mean().round(3))
print()
for style in sorted(df.blue_style.unique()):
    s = df[df.blue_style == style]
    trig = s["split_detector_first_trigger_step"] >= 0
    mean_first = (
        float(s.loc[trig, "split_detector_first_trigger_step"].mean()) if trig.any() else float("nan")
    )
    print(
        f"{style}: n={len(s)} trigger={trig.mean():.2f} first={mean_first:.1f} "
        f"margin={s.win_margin.mean():.3f}"
    )
    cols = [
        "episode_index",
        "win_margin",
        "blue_score",
        "red_score",
        "split_detector_first_trigger_step",
        "split_detector_active_steps",
        "steps",
    ]
    print(s[cols].to_string(index=False))
    print()
