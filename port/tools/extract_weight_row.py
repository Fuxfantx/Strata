"""Extract one f32 weight row from a GGUF (dequantising q4_0 in numpy) so a C++ codec check can use real data.

Random N(0,1) rows made the trellis search pathologically slow, and reading GGUF from C++ is where the last attempt
stalled; numpy already has the q4_0 decoder here.
"""
import struct
import sys

import numpy as np

PATH = r"C:\Users\Fuxfantx\C-Res\Qwen\3.8-27B-Q4\Qwen3.8-27B-Q4_0.gguf"
TENSOR = "blk.12.attn_qkv.weight"
OUT = sys.argv[1] if len(sys.argv) > 1 else r"build\quant\row_attn.f32"


def read_gguf_tensor(path, name):
    with open(path, "rb") as f:
        data = f.read()
    off = 4
    version, = struct.unpack_from("<I", data, off); off += 4
    n_tensors, n_kv = struct.unpack_from("<QQ", data, off); off += 16

    def rstr(o):
        n, = struct.unpack_from("<Q", data, o)
        o += 8
        return data[o:o + n].decode("utf-8", "replace"), o + n

    fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    alignment = 32
    for _ in range(n_kv):
        k, off = rstr(off)
        t, = struct.unpack_from("<I", data, off); off += 4
        if t == 8:
            _, off = rstr(off)
        elif t == 9:
            at, = struct.unpack_from("<I", data, off); off += 4
            n, = struct.unpack_from("<Q", data, off); off += 8
            if at == 8:
                for _ in range(n):
                    _, off = rstr(off)
            else:
                off += n * fixed[at]
        else:
            if k == "general.alignment":
                alignment, = struct.unpack_from("<I", data, off)
            off += fixed[t]

    tensors = {}
    for _ in range(n_tensors):
        nm, off = rstr(off)
        nd, = struct.unpack_from("<I", data, off); off += 4
        dims = struct.unpack_from("<" + "Q" * nd, data, off); off += 8 * nd
        ttype, = struct.unpack_from("<I", data, off); off += 4
        toff, = struct.unpack_from("<Q", data, off); off += 8
        tensors[nm] = (dims, ttype, toff)

    data_off = (off + alignment - 1) // alignment * alignment
    dims, ttype, toff = tensors[name]
    return data, data_off + toff, dims, ttype


def dequant_q4_0(raw, ne):
    blocks = ne // 32
    out = np.empty(ne, dtype=np.float32)
    for b in range(blocks):
        d, = struct.unpack_from("<e", raw, b * 18)
        qs = np.frombuffer(raw, dtype=np.uint8, count=16, offset=b * 18 + 2)
        lo = (qs & 0x0F).astype(np.int8) - 8
        hi = (qs >> 4).astype(np.int8) - 8
        vals = np.empty(32, dtype=np.float32)
        vals[0::2] = lo
        vals[1::2] = hi
        out[b * 32:(b + 1) * 32] = vals * d
    return out


data, off, dims, ttype = read_gguf_tensor(PATH, TENSOR)
ne0 = dims[0]
row_bytes = (ne0 // 32) * 18 if ttype == 2 else None
print(f"{TENSOR}: dims {dims} type {ttype} ne0 {ne0} row_bytes {row_bytes}")
raw = data[off:off + row_bytes]
row = dequant_q4_0(raw, ne0) if ttype == 2 else None
if row is None:
    print("this script only dequantises q4_0")
    sys.exit(1)
row.astype("<f4").tofile(OUT)
print(f"wrote {ne0} floats to {OUT}: mean {row.mean():.6f} std {row.std():.6f}")
