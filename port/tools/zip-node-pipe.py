"""Pair the node and pipe log lines by position and report which node names issued which id pipeline.

The histogram alone cannot say whether matmul_id_coop belonged to the 2-bit experts or to the bf16 tensors, because it
only carries pipeline names.  STRATA_VK_NODE_LOG and STRATA_VK_PIPE_LOG emit one line each per dispatch in order, so
zipping them answers it.
"""
import collections
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
patterns = sys.argv[2].split(',') if len(sys.argv) > 2 else ['matmul_id_coop', 'mul_mat_vec_id']

lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
pairs = []
last_node = None
for line in lines:
    if line.startswith('[node]'):
        last_node = line
    elif line.startswith('[pipe]'):
        if last_node is not None:
            pairs.append((last_node, line))

for pat in patterns:
    hit = [(n, p) for n, p in pairs if pat in p]
    print(f'{pat}: {len(hit)} dispatches')
    seen = collections.Counter()
    for n, p in hit:
        name = n.split(' ', 3)[-1] if n.count(' ') >= 3 else n
        seen[name] += 1
    for name, count in seen.most_common(6):
        print(f'  {count:5d}  {name}')
    for n, p in hit[:3]:
        print(f'    e.g. {n.strip()}  ->  {p.strip()}')

print(f'total pairs {len(pairs)}')
