"""Print name and type id per tensor for packs whose type ids gguf-py cannot decode (the fork's trellis types)."""
import struct
import sys

TYPE_NAMES = {0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0", 9: "q8_1",
              10: "q2_k", 11: "q3_k", 12: "q4_k", 13: "q5_k", 14: "q6_k", 30: "iq4_xs", 41: "q2_0", 42: "q1_0",
              43: "ptq1_0", 44: "tq2_t"}


def read(path):
    with open(path, "rb") as f:
        data = f.read()
    off = 4
    off += 4
    n_tensors, n_kv = struct.unpack_from("<QQ", data, off)
    off += 16

    def rstr(o):
        n, = struct.unpack_from("<Q", data, o)
        o += 8
        return data[o:o + n].decode("utf-8", "replace"), o + n

    fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    for _ in range(n_kv):
        _, off = rstr(off)
        t, = struct.unpack_from("<I", data, off)
        off += 4
        if t == 8:
            _, off = rstr(off)
        elif t == 9:
            at, = struct.unpack_from("<I", data, off)
            off += 4
            n, = struct.unpack_from("<Q", data, off)
            off += 8
            if at == 8:
                for _ in range(n):
                    _, off = rstr(off)
            else:
                off += n * fixed[at]
        else:
            off += fixed[t]

    out = []
    for _ in range(n_tensors):
        name, off = rstr(off)
        nd, = struct.unpack_from("<I", data, off)
        off += 4
        dims = struct.unpack_from("<" + "Q" * nd, data, off)
        off += 8 * nd
        ttype, = struct.unpack_from("<I", data, off)
        off += 4 + 8
        out.append((name, int(ttype), dims))
    return out


for path in sys.argv[1:]:
    print("===", path)
    for name, ttype, dims in read(path):
        if any(k in name for k in ("token_embd", "output", "attn_qkv", "ffn_down", "ple")):
            print(f"  {name:40s} {TYPE_NAMES.get(ttype, 'type' + str(ttype)):8s} {list(dims)}")
