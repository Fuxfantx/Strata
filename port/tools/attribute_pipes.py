"""Attribute the [pipe] lines of STRATA_VK_PIPE_LOG to the nodes that issued them.

The log interleaves one [node] marker per graph node with the [pipe] line(s) that node dispatched, so zipping them
gives "which graph node used which pipeline".  What matters for a dispatch-count campaign is the opposite direction:
for each pipeline, which node families issue it, so the ones that can be fused away are visible by name.
"""
import collections
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'build/nl_err.log'
node_re = re.compile(r'^\[node\] (\d+) (\S+) (.*)$')
pipe_re = re.compile(r'^\[pipe\] (\S+) ')

rows = []
cur = None
with open(path, encoding='utf-8', errors='replace') as f:
    for line in f:
        m = node_re.match(line)
        if m:
            cur = (int(m.group(1)), m.group(2), m.group(3).strip())
            continue
        m = pipe_re.match(line)
        if m and cur is not None:
            rows.append((cur[1], cur[2], m.group(1)))
            cur = None

print('attributed dispatches:', len(rows))

# which pipelines does each op family use
by_op = collections.defaultdict(collections.Counter)
for op, name, pipe in rows:
    by_op[op][pipe] += 1
print('\n=== per op, top pipelines ===')
for op in sorted(by_op, key=lambda o: -sum(by_op[o].values())):
    pipes = by_op[op]
    total = sum(pipes.values())
    top = ', '.join(f'{p}:{n}' for p, n in pipes.most_common(4))
    print(f'{total:6d}  {op:12s}  {top}')

# for the elementwise families, which node names issue them
targets = sys.argv[2].split(',') if len(sys.argv) > 2 else \
    ['SCALE', 'MUL', 'ADD', 'CONCAT', 'CONT', 'SILU', 'SIGMOID', 'GLU', 'FWHT']
for target in targets:
    names = collections.Counter()
    for op, name, pipe in rows:
        if op == target:
            names[name] += 1
    if not names:
        continue
    print(f'\n=== {target}: node names (top 12 of {sum(names.values())}) ===')
    for n, c in names.most_common(12):
        print(f'{c:6d}  {n}')
