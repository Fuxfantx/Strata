"""Compare two engine eval dumps (CPU vs Vulkan) node by node and print the first divergence."""
import re
import sys

PAT = re.compile(r"\[eval\]\s+(\S+)\s+(\S+)\s+n=(\d+)\s+sum=(\S+)\s+abs=(\S+)\s+lo=(\S+)\s+hi=(\S+)")


def load(path):
    out = {}
    order = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = PAT.search(line)
        if not m:
            continue
        name, op, n = m.group(1), m.group(2), int(m.group(3))
        vals = [float(m.group(i)) for i in (4, 5, 6, 7)]
        if name in out:          # keep the first graph execution
            continue
        out[name] = (op, n, vals)
        order.append(name)
    return out, order


a, oa = load(sys.argv[1])
b, ob = load(sys.argv[2])
print(f"cpu nodes: {len(a)}   vk nodes: {len(b)}")
common = [x for x in oa if x in b]
print(f"common: {len(common)}")

# index by node number so that order differences do not matter
def idx(name):
    m = re.fullmatch(r"node_(\d+)", name)
    return int(m.group(1)) if m else None

first_bad = None
bad = []
for name in common:
    op, n, va = a[name]
    _, _, vb = b[name]
    scale = max(abs(va[1]), 1e-3)
    dsum = abs(va[0] - vb[0])
    dabs = abs(va[1] - vb[1])
    rel = max(dsum, dabs) / scale
    if rel > 1e-4:
        bad.append((idx(name), name, op, n, rel, va, vb))

bad.sort(key=lambda x: (x[0] if x[0] is not None else 10**9))
print(f"divergent nodes: {len(bad)}")
for row in bad[:10]:
    i, name, op, n, rel, va, vb = row
    print(f"  {name:>10} {op:<12} n={n:<6} rel={rel:.3e}  cpu(abs={va[1]:.6g}) vk(abs={vb[1]:.6g})")
