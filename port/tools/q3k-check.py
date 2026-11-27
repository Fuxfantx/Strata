#!/usr/bin/env python3
"""Decode one Q3_K block independently and compare it with ggml's own dequantiser.

The Vulkan kernel's Q3_K path disagreed with the CPU reference, and debugging that through a 40 GiB model is the wrong
loop.  This tool takes the same 110-byte block the shader reads, decodes it here from the format definition, and
compares against the row `latent-probe --dump-row` prints through `ggml_get_type_traits(Q3_K)->to_float`.  When the two
agree, the format understanding is right and any remaining difference is in the shader; when they disagree, the report
says which sub-block and which value.

    python port/tools/q3k-check.py --dump q3k-row.txt --file layer0.gguf --offset 77990496
"""
import argparse
import pathlib
import struct
import sys

import numpy as np

QK_K = 256
BLOCK_BYTES = 110


def half(b):
    return struct.unpack('<e', b)[0]


def decode_block(raw: bytes) -> np.ndarray:
    assert len(raw) == BLOCK_BYTES
    hmask = np.frombuffer(raw[0:32], dtype=np.uint8)
    qs = np.frombuffer(raw[32:96], dtype=np.uint8)
    scales_raw = raw[96:108]
    d_all = half(raw[108:110])

    # six-bit scales: 12 bytes hold sixteen values, four per byte split as low nibbles and the extra high bits in the
    # last four bytes - the same unpacking ggml does before it subtracts 32
    aux = list(struct.unpack('<III', scales_raw))
    tmp = aux[2]
    km1, km2 = 0x03030303, 0x0F0F0F0F
    a0 = (aux[0] & km2) | (((tmp >> 0) & km1) << 4)
    a1 = (aux[1] & km2) | (((tmp >> 2) & km1) << 4)
    a2 = ((aux[0] >> 4) & km2) | (((tmp >> 4) & km1) << 4)
    a3 = ((aux[1] >> 4) & km2) | (((tmp >> 6) & km1) << 4)
    scales = []
    for word in (a0, a1, a2, a3):
        scales.extend([(word >> (8 * i)) & 0xFF for i in range(4)])
    scales = [s - 32 for s in scales]

    out = np.zeros(QK_K, dtype=np.float32)
    is_ = 0
    m = 1
    qi = 0
    for n in range(0, QK_K, 128):
        shift = 0
        for j in range(4):
            for l in range(16):
                q2 = (qs[qi + l] >> shift) & 3
                bit = 0 if (hmask[l] & m) else 4
                out[n + j * 32 + l] = d_all * scales[is_] * (int(q2) - bit)
            is_ += 1
            for l in range(16):
                q2 = (qs[qi + l + 16] >> shift) & 3
                bit = 0 if (hmask[l + 16] & m) else 4
                out[n + j * 32 + 16 + l] = d_all * scales[is_] * (int(q2) - bit)
            is_ += 1
            shift += 2
            m <<= 1
        qi += 32
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dump', required=True, help='the file latent-probe --dump-row wrote')
    ap.add_argument('--file', required=True, help='the file the row came from')
    ap.add_argument('--offset', type=int, required=True)
    args = ap.parse_args()

    lines = pathlib.Path(args.dump).read_text(encoding='utf-8').splitlines()
    ref = np.array([float(v) for v in lines[1].split() + [x for l in lines[2:] for x in l.split()]], dtype=np.float32)
    print('reference values from ggml:', ref.size)

    # a row can be several blocks long (latent 1792 is seven), so decode the whole row
    n_blocks = max(1, ref.size // QK_K)
    with open(args.file, 'rb') as f:
        f.seek(args.offset)
        raw = f.read(BLOCK_BYTES * n_blocks)
    got = np.concatenate([decode_block(raw[i * BLOCK_BYTES:(i + 1) * BLOCK_BYTES]) for i in range(n_blocks)])

    diff = np.abs(got - ref)
    print('max abs difference %.6g at index %d (got %.6g, reference %.6g)'
          % (diff.max(), int(diff.argmax()), got[diff.argmax()], ref[diff.argmax()]))
    print('first eight  got:', np.round(got[:8], 4))
    print('first eight  ref:', np.round(ref[:8], 4))
    bad = diff > 1e-6
    if bad.any():
        idx = np.flatnonzero(bad)
        print('mismatching values: %d of %d, first at %s' % (bad.sum(), bad.size, idx[:8]))
        for i in idx[:8]:
            print('  [%3d] sub-block %2d: got %.6g  ref %.6g' % (i, i // 16, got[i], ref[i]))
        return 1
    print('decoder agrees with ggml')
    return 0


if __name__ == '__main__':
    sys.exit(main())
