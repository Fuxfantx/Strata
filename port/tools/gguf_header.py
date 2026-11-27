#!/usr/bin/env python3
"""Print every GGUF key/value pair and every tensor name, tolerating type ids this gguf-py does not know.

`gguf_types.py` answers "what is in this pack"; this one answers "what does this pack declare", which is what a diff
between two packs needs (a fork type such as TQ2_T makes gguf-py's reader fail, and the tensor names are enough to see
whether two packs hold the same model).

    python port/tools/gguf_header.py model.gguf [more.gguf ...]

With two or more files it prints the differences instead of the full dump.
"""
import struct
import sys


def read(path):
    with open(path, "rb") as f:
        data = f.read(64 * 1024 * 1024)  # header + metadata + tensor infos only
    if data[:4] != b"GGUF":
        raise ValueError(f"{path}: not a GGUF file")
    off = 4
    version, = struct.unpack_from("<I", data, off)
    off += 4
    n_tensors, n_kv = struct.unpack_from("<QQ", data, off)
    off += 16

    def rstr(o):
        n, = struct.unpack_from("<Q", data, o)
        o += 8
        return data[o:o + n].decode("utf-8", "replace"), o + n

    fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    kv = {}
    for _ in range(n_kv):
        k, off = rstr(off)
        t, = struct.unpack_from("<I", data, off)
        off += 4
        if t == 8:
            v, off = rstr(off)
        elif t == 9:
            at, = struct.unpack_from("<I", data, off)
            off += 4
            n, = struct.unpack_from("<Q", data, off)
            off += 8
            if at == 8:
                for _ in range(n):
                    _, off = rstr(off)
                v = f"<str[{n}]>"
            else:
                off += n * fixed[at]
                v = f"<type{at}[{n}]>"
        else:
            raw = data[off:off + fixed[t]]
            off += fixed[t]
            v = raw.hex() if t == 7 else int.from_bytes(raw, "little")
        kv[k] = v

    names = []
    for _ in range(n_tensors):
        n, off = rstr(off)
        nd, = struct.unpack_from("<I", data, off)
        off += 4 + 8 * nd + 4 + 8
        names.append(n)
    return version, kv, names


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip())
        return 1
    packs = [read(p) for p in argv[1:]]
    if len(packs) == 1:
        version, kv, names = packs[0]
        print(f"gguf version {version}, {len(kv)} key/value pairs, {len(names)} tensors")
        for k in sorted(kv):
            print(f"  {k} = {kv[k]}")
        for n in names:
            print(f"  tensor {n}")
        return 0

    base_version, base_kv, base_names = packs[0]
    for path, (version, kv, names) in zip(argv[2:], packs[1:]):
        print(f"=== {argv[1]} vs {path} ===")
        for k in sorted(set(base_kv) - set(kv)):
            print(f"  only in {argv[1]}: {k} = {base_kv[k]}")
        for k in sorted(set(kv) - set(base_kv)):
            print(f"  only in {path}: {k} = {kv[k]}")
        for k in sorted(set(kv) & set(base_kv)):
            if kv[k] != base_kv[k]:
                print(f"  {k}: {base_kv[k]!r} vs {kv[k]!r}")
        only_a = sorted(set(base_names) - set(names))
        only_b = sorted(set(names) - set(base_names))
        if only_a:
            print(f"  tensors only in {argv[1]}: {only_a[:10]}")
        if only_b:
            print(f"  tensors only in {path}: {only_b[:10]}")
        if not only_a and not only_b:
            print(f"  the same {len(names)} tensor names")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
