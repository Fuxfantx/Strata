"""Count the decode graph's nodes by op, so the dispatch budget is a fact rather than a guess.

STRATA_VK_PIPE_LOG=1 prints one [node] line per dispatched node (with the fused chain after it).  The graphs are separated
by the node index resetting, and the decode graph is the one with the most nodes among the small ones.
"""
import collections
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else 'nodes.err')
lines = path.read_text(encoding='utf-8', errors='replace').splitlines()

graphs, cur, last = [], [], -1
for line in lines:
    m = re.match(r'\[node\] (\d+) (\S+) (\S*)(.*)', line)
    if not m:
        continue
    idx = int(m.group(1))
    if idx < last:
        graphs.append(cur)
        cur = []
    last = idx
    fused = [s for s in m.group(4).split(',') if s]
    cur.append((m.group(2), m.group(3), len(fused)))
if cur:
    graphs.append(cur)

print('graphs:', [len(g) for g in graphs])
if not graphs:
    sys.exit('no nodes found')

# the decode graph: the last graph with the most dispatches among graphs of a few thousand nodes
g = max(graphs, key=len)
print('largest graph:', len(g), 'dispatches')

counts = collections.Counter(op for op, _, _ in g)
fused = sum(f for _, _, f in g)
print(f'fused follow-ons inside those dispatches: {fused}')
print('by op:')
for op, n in counts.most_common(24):
    print(f'  {n:5d}  {op}')

# what the per-op dispatch count implies: a barrier per dispatch at ~3.5 us
print()
print(f'{len(g)} dispatches x 3.5 us = {len(g) * 3.5 / 1000:.2f} ms of barrier cost')
