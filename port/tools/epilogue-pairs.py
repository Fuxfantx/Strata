"""Which elementwise ops sit immediately after a matmul in the decode graph, by frequency.

The dispatch budget says MUL_MAT 881, SCALE 281, RMS_NORM 184, TOP_K_MASK 175, ADD 173, MUL 149, GLU 96.  This looks at the
recorded order and counts, for every dispatch that is a matmul, the op of the next dispatch - which is the fusion candidate
and the number of barriers it would save.
"""
import collections
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else 'nodes.err')
lines = path.read_text(encoding='utf-8', errors='replace').splitlines()

graphs, cur, last = [], [], -1
for line in lines:
    m = re.match(r'\[node\] (\d+) (\S+) (\S*)', line)
    if not m:
        continue
    idx = int(m.group(1))
    if idx < last:
        graphs.append(cur)
        cur = []
    last = idx
    cur.append((m.group(2), m.group(3)))
if cur:
    graphs.append(cur)

g = max(graphs, key=len)
print(f'decode graph: {len(g)} dispatches')

pairs = collections.Counter()
for (op_a, _), (op_b, _) in zip(g, g[1:]):
    if op_a in ('MUL_MAT', 'MUL_MAT_ID'):
        pairs[(op_a, op_b)] += 1

print('matmul -> next op:')
for (a, b), n in pairs.most_common(20):
    print(f'  {n:5d}  {a} -> {b}')

after_mm = sum(n for (a, b), n in pairs.items() if b not in ('MUL_MAT', 'MUL_MAT_ID'))
print(f'\nmatmuls followed by a non-matmul: {after_mm}  (each is a candidate epilogue fusion)')
print(f'at 3.5 us per saved barrier that is up to {after_mm * 3.5 / 1000:.2f} ms')
