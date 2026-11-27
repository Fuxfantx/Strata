"""Dump a GGUF's key metadata that mentions ple or engram (arrives at the PLE sizing question)."""
import struct
import sys

FIXED = {0: ('<B', 1), 1: ('<b', 1), 2: ('<H', 2), 3: ('<h', 2), 4: ('<I', 4), 5: ('<i', 4),
         6: ('<f', 4), 7: ('<?', 1), 10: ('<Q', 8), 11: ('<q', 8), 12: ('<d', 8)}


def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')


def read_val(f, t):
    if t in FIXED:
        fmt, size = FIXED[t]
        return struct.unpack(fmt, f.read(size))[0]
    if t == 8:
        return read_str(f)
    if t == 9:
        at, n = struct.unpack('<IQ', f.read(12))
        return [read_val(f, at) for _ in range(n)]
    raise ValueError(f'unknown kv type {t}')


f = open(sys.argv[1], 'rb')
magic, version, n_tensors, n_kv = struct.unpack('<IIQQ', f.read(24))
assert magic == 0x46554747
for _ in range(n_kv):
    key = read_str(f)
    t = struct.unpack('<I', f.read(4))[0]
    v = read_val(f, t)
    if 'ple' in key.lower() or 'engram' in key.lower():
        print(f'{key} = {str(v)[:200]}')
