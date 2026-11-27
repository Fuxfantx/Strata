import pathlib
import re
import sys

PAT = re.compile(r'^\[eval\]\s+(\S+)\s+(\S+)\s+n=(-?\d+)\s+abs=(\S+)\s*$')


def load(path):
    """Return [(name, op, n, abs, occurrence)] in order."""
    seen = {}
    out = []
    for line in pathlib.Path(path).read_text(encoding='utf-8', errors='surrogateescape').splitlines():
        m = PAT.match(line.strip())
        if not m:
            continue
        name, op, n, ab = m.group(1), m.group(2), int(m.group(3)), float(m.group(4))
        if op == 'VIEW':
            continue
        key = (name, op, n)
        occ = seen.get(key, 0)
        seen[key] = occ + 1
        out.append((name, op, n, ab, occ))
    return out


cpu = load(sys.argv[1])
vk = load(sys.argv[2])
vk_map = {(name, op, n, occ): ab for (name, op, n, ab, occ) in vk}

threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-2

first = None
context = []
for i, (name, op, n, ab, occ) in enumerate(cpu):
    key = (name, op, n, occ)
    if key not in vk_map:
        continue
    vab = vk_map[key]
    rel = abs(vab - ab) / max(abs(ab), 1.0)
    if rel > threshold and first is None:
        first = (i, name, op, n, occ, ab, vab, rel)
        break
    context.append((i, name, op, ab, vab, rel))
    if len(context) > 5:
        context.pop(0)

if first is None:
    print('no divergence above %.3g in the nodes both dumps share' % threshold)
else:
    i, name, op, n, occ, cab, vab, rel = first
    print('first divergence (cpu order) at index %d: node=%s op=%s n=%d occ=%d' % (i, name, op, n, occ))
    print('  cpu abs=%.9g   vk abs=%.9g   rel=%.3g' % (cab, vab, rel))
    print('  preceding matched nodes:')
    for ci, cname, cop, cab2, vab2, crel in context:
        print('    %6d %-28s %-14s cpu=%.9g vk=%.9g rel=%.2g' % (ci, cname, cop, cab2, vab2, crel))
