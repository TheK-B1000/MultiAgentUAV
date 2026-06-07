import csv
rows = list(csv.DictReader(open('checkpoints/4v4/latent_v3i15_strong_separation_50k_test_4v4_episodes.csv')))
print(f"Episodes: {len(rows)}")
if rows:
    print(f"Last row keys: {list(rows[-1].keys())[:5]}")
    gs = rows[-1].get("global_step", rows[-1].get("step", "?"))
    print(f"Last step: {gs}")
    print(f"Last 3 rows:")
    for r in rows[-3:]:
        print({k: r[k] for k in list(r.keys())[:6]})
