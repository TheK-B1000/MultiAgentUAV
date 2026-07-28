import csv, collections, sys
path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
g = collections.defaultdict(list)
w = collections.defaultdict(int)
for r in rows:
    g[r["blue_style"]].append(float(r["win_margin"]))
    w[r["blue_style"]] += int(r["success"])
print("n=", len(rows))
for k in sorted(g):
    v = g[k]
    print(f"{k}: n={len(v)} WR={w[k]}/{len(v)} mean={sum(v)/len(v):.3f}")
