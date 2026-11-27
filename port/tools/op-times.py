"""Aggregate the Vulkan per-node GPU timings, per graph block, and report the decode graph's op table.

GGML_VK_PERF_LOGGER=1 prints one block per graph_compute, ending with a Total time line.  The timestamps are only
comparable inside one block (the query pool is reset per graph), so the blocks are split first and the largest one - a
decode step - is the table worth reading.
"""
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else 'perf.err')
text = path.read_text(encoding='utf-8', errors='replace')

blocks, cur = [], []
for line in text.splitlines():
    cur.append(line)
    if 'Total time' in line:
        blocks.append(cur)
        cur = []
if cur:
    blocks.append(cur)

op_re = re.compile(r'^\s*(.+?):\s*(\d+) x ([\d.]+) us = ([\d.]+) us')
tot_re = re.compile(r'Total time:\s*([\d.]+) us')

print(f'blocks: {len(blocks)}')
parsed = []
for i, b in enumerate(blocks):
    ops = [op_re.match(l) for l in b]
    ops = [m for m in ops if m]
    tot = None
    for l in b:
        m = tot_re.search(l)
        if m:
            tot = float(m.group(1))
    parsed.append((tot or 0.0, ops))
    print(f'  block {i}: {len(ops):4d} op rows, total {tot} us')

# the decode graph is the block with the most op rows (a decode step has ~2700 nodes)
parsed.sort(key=lambda x: len(x[1]), reverse=True)
tot, ops = parsed[0]
print()
print(f'decode graph: {len(ops)} op rows, GPU total {tot:.0f} us')
by_class = {}
for m in ops:
    name = m.group(1)
    us = float(m.group(4))
    n = int(m.group(2))
    key = re.sub(r'\s+m=.*$', '', name).strip()
    e = by_class.setdefault(key, [0.0, 0, name])
    e[0] += us
    e[1] += n
rows = sorted(((us, n, name) for us, n, name in by_class.values()), reverse=True)
print(f'{"us":>9} {"n":>5} {"per-us":>8} {"%gpu":>6}  op')
acc = 0.0
for us, n, name in rows[:20]:
    acc += us
    print(f'{us:9.0f} {n:5d} {us / n:8.2f} {100 * us / tot if tot else 0:6.1f}  {name[:84]}')
print(f'top 20 = {acc:.0f} us = {100 * acc / tot if tot else 0:.1f}% of the step; sum of all = {sum(r[0] for r in rows):.0f} us')
