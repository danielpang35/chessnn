from collections import Counter

from nnue.dataset import iter_fen_cp_rows

path = "data/raw/positions.csv"

n = 0
cp = []

for fen, y_cp in iter_fen_cp_rows(path):
    n += 1
    cp.append(int(y_cp))

print("rows:", n)
print("cp min/max:", min(cp), max(cp))
print("cp mean:", sum(cp) / len(cp))

b = Counter((c // 50) * 50 for c in cp)
print("most common buckets:", b.most_common(10))
