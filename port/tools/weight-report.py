"""Report a pack's decode traffic: bytes per token, the tensors that dominate it, and the implied bandwidth.

Sizes come from the tensor offset table in the header, not from a type table, so this reads packs whose types the
vendored gguf-py does not know yet (TQK6/TQK7 and the rest of the port's family).
"""
import struct
import sys

TYPE_NAMES = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1',
              10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K',
              16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S',
              22: 'IQ2_S', 23: 'IQ4_XS', 24: 'I8', 25: 'I16', 26: 'I32', 27: 'I64', 28: 'F64',
              30: 'BF16', 34: 'TQ1_0', 35: 'TQ2_0', 39: 'MXFP4', 40: 'Q1_0', 41: 'Q2_0_FORK', 42: 'Q2_0',
              143: 'PTQ1_0', 144: 'TQ2_T', 145: 'TQK6', 146: 'TQK7'}


def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')


def skip(f, t):
    fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in fixed:
        f.read(fixed[t])
    elif t == 8:
        read_str(f)
    elif t == 9:
        at, n = struct.unpack('<IQ', f.read(12))
        for _ in range(n):
            skip(f, at)
    else:
        raise ValueError(f'unknown kv type {t}')


def main(path, tok_s=None):
    f = open(path, 'rb')
    magic, version, n_tensors, n_kv = struct.unpack('<IIQQ', f.read(24))
    assert magic == 0x46554747, 'not a gguf'
    meta = {}
    for _ in range(n_kv):
        k = read_str(f)
        t = struct.unpack('<I', f.read(4))[0]
        if t in (4, 5):
            meta[k] = struct.unpack('<I' if t == 4 else '<i', f.read(4))[0]
        elif t == 8:
            meta[k] = read_str(f)
        else:
            skip(f, t)

    infos = []
    for _ in range(n_tensors):
        name = read_str(f)
        nd = struct.unpack('<I', f.read(4))[0]
        dims = struct.unpack('<' + 'Q' * nd, f.read(8 * nd))
        tt = struct.unpack('<I', f.read(4))[0]
        off = struct.unpack('<Q', f.read(8))[0]
        infos.append([name, tt, off, dims])

    # the last tensor ends at its own size, which the header does not store: derive every size from the next offset and
    # give the last one the same bytes-per-element as the second largest of its type (close enough for a budget)
    infos.sort(key=lambda r: r[2])
    sizes = []
    for i, (name, tt, off, dims) in enumerate(infos):
        if i + 1 < len(infos):
            sizes.append(infos[i + 1][2] - off)
        else:
            sizes.append(None)
    known = [s for s in sizes if s]
    median_tail = known[-1] if known else 0
    sizes[-1] = median_tail

    dense = experts = 0
    per_type = {}
    rows = []
    for (name, tt, off, dims), size in zip(infos, sizes):
        is_expert = '_exps' in name
        per_type[TYPE_NAMES.get(tt, f'type{tt}')] = per_type.get(TYPE_NAMES.get(tt, f'type{tt}'), 0) + size
        if is_expert:
            experts += size
        else:
            dense += size
        rows.append((size, name, TYPE_NAMES.get(tt, f'type{tt}')))

    used = int(meta.get('qwen4exp.expert_used_count', meta.get('qwen35moe.expert_used_count', 1)))
    total = int(meta.get('qwen4exp.expert_count', meta.get('qwen35moe.expert_count', 1)))
    print(path)
    print(f'  arch {meta.get("general.architecture")}  blocks {meta.get("qwen4exp.block_count")}  '
          f'experts {used}/{total}  d={meta.get("qwen4exp.embedding_length")}')
    print(f'  headers: dense {dense / 2**30:.2f} GiB   experts {experts / 2**30:.2f} GiB')
    print('  by type (GiB): ' + ', '.join(f'{k} {v / 2**30:.2f}' for k, v in sorted(per_type.items(), key=lambda kv: -kv[1])[:8]))
    share = used / max(total, 1)
    per_token = dense + experts * share
    print(f'  weights read per token: {per_token / 2**30:.2f} GiB  ({per_token / 1e9:.2f} GB)')
    if tok_s:
        print(f'  at {tok_s} tok/s -> {per_token * tok_s / 1e9:.1f} GB/s sustained; at 140 GB/s -> {140e9 / per_token:.1f} tok/s')
    rows.sort(reverse=True)
    print('  largest tensors:')
    for size, name, tt in rows[:8]:
        print(f'    {size / 2**20:9.1f} MiB  {tt:8s} {name}')


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else None)
