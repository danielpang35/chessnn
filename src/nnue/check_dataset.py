import csv
from collections import Counter

path = "data/raw/positions.csv"

n = 0
cp = []

with open(path, newline="", encoding="utf-8") as f:
    r = csv.reader(f)
    header = next(r, None)  # skip header if present
    for row in r:
        if not row or len(row) < 2:
            continue
        try:
            cp_val = int(row[1])
        except ValueError:
            continue
        n += 1
        cp.append(cp_val)

print("rows:", n)
print("cp min/max:", min(cp), max(cp))
print("cp mean:", sum(cp) / len(cp))

b = Counter((c // 50) * 50 for c in cp)
print("most common buckets:", b.most_common(10))
