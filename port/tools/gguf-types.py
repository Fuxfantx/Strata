"""Print the tensor type histogram of a GGUF pack.

The type ids past BF16 are fork specific, so an unknown id is printed as typeN.  Used to decide which repack is
possible: a conversion has to be lossless, so 2 bit weights can go to q4_0 but 8 bit and k quant weights cannot.

Usage: python port/tools/gguf-types.py <pack.gguf> [more.gguf ...]
"""

import struct
import sys
import pathlib

NAMES = {
    0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1',
    10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K',
    16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S',
    22: 'IQ2_S', 23: 'IQ4_XS', 24: 'I8', 25: 'I16', 26: 'I32', 27: 'I64', 28: 'F64',
    29: 'IQ1_M', 30: 'BF16', 34: 'TQ1_0', 35: 'TQ2_0', 39: 'MXFP4', 40: 'Q2_0', 41: 'PTQ1_0',
}

# GGUF metadata value types: uint8, int8, uint16, int16, uint32, int32, float32, bool, string, array,
# uint64, int64, float64.
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    if n > (1 << 24):
        raise SystemExit('implausible string length %d, parser out of step' % n)
    return f.read(n).decode('utf-8', 'replace')


def skip_value(f, t):
    if t in SCALAR:
        f.read(SCALAR[t])
    elif t == 8:
        read_str(f)
    elif t == 9:
        et = struct.unpack('<I', f.read(4))[0]
        cnt = struct.unpack('<Q', f.read(8))[0]
        if et in SCALAR:
            f.read(SCALAR[et] * cnt)
        else:
            for _ in range(cnt):
                skip_value(f, et)
    else:
        raise SystemExit('unknown kv type %d' % t)


def hist(path):
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            raise SystemExit('not a gguf: ' + path)
        ver = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n_kv):
            read_str(f)
            skip_value(f, struct.unpack('<I', f.read(4))[0])
        counts = {}
        for _ in range(n_tensors):
            read_str(f)
            nd = struct.unpack('<I', f.read(4))[0]
            f.read(8 * nd)
            t = struct.unpack('<I', f.read(4))[0]
            f.read(8)
            counts[t] = counts.get(t, 0) + 1
    print(pathlib.Path(path).name, 'v%d' % ver, 'tensors', n_tensors)
    for t, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print('   %-8s %5d' % (NAMES.get(t, 'type%d' % t), c))


for arg in sys.argv[1:]:
    hist(arg)
