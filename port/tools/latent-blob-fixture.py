#!/usr/bin/env python3
"""A tiny latent pack with the released packs' expert layout, for testing the blob path.

The released 145B pack's expert blob is 1.6 MB per expert and 512 experts per layer, which is too large to upload for a
single unit test.  This writes a pack with the same layout - Q3_K gate/up, IQ4_NL down, one blob per expert - at a size
where both backends can be compared in a second:

    latent 256 (a whole Q3_K block), n_ff 64 (a whole IQ4_NL row of blocks), n_embd 128, 2 groups, 4 experts, 1 layer

The quantised payload is random bytes: the test is that the CPU reference and the kernel read the same bytes the same
way, not that the values mean anything.

    python port/tools/latent-blob-fixture.py OUTDIR
"""
import json
import pathlib
import struct
import sys

import numpy as np

H, LATENT, GROUPS, NFF, E, L = 128, 256, 2, 64, 4, 1


def q8_blocks(rows, n_values, rng):
    """Q8_0 rows: one half scale and 32 int8 per block."""
    out = bytearray()
    for _ in range(rows):
        codes = rng.integers(-127, 128, size=n_values).astype(np.int8)
        for blk in range(n_values // 32):
            out += struct.pack('<e', 1.0 / 128.0)
            out += codes[blk * 32:(blk + 1) * 32].tobytes()
    return bytes(out)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = pathlib.Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)

    (root / 'model.json').write_text(json.dumps({
        'model_name': 'blob-fixture', 'family': 'fixture', 'layers': L, 'experts': E,
        'latent_dim': LATENT, 'latent_groups': GROUPS, 'embedding_length': H,
        'build': 'Q3_K/IQ4_NL', 'pack_format': 'strata pack index v3',
    }, indent=2), encoding='utf-8')

    labels = rng.integers(0, GROUPS, size=(L, E)).astype(np.int32)
    (root / 'labels.bin').write_bytes(labels.tobytes())

    with (root / 'latent_dense_q8.bin').open('wb') as f:
        for _ in range(L):
            f.write(q8_blocks(GROUPS * LATENT, H, rng))      # D: one row per (group, latent)
            f.write(q8_blocks(GROUPS * H, LATENT, rng))      # U: one row per (group, embd)

    gu_row = (LATENT // 256) * 110
    dn_row = (NFF // 32) * 18
    blob = 2 * NFF * gu_row + LATENT * dn_row

    def q3k_payload(n_blocks):
        """Random codes, but a finite scale: random bytes in the scale field decode to NaN halves often enough that the
        whole row would be NaN, and a comparison of NaNs proves nothing."""
        out = bytearray()
        for _ in range(n_blocks):
            out += rng.integers(0, 256, size=96, dtype=np.uint8).tobytes()      # hmask and qs
            out += rng.integers(0, 256, size=12, dtype=np.uint8).tobytes()      # six-bit scales
            out += struct.pack('<e', 0.0625)                                    # d
        return bytes(out)

    def iq4_payload(n_blocks):
        out = bytearray()
        for _ in range(n_blocks):
            out += struct.pack('<e', 0.0625)                                    # d
            out += rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()      # codes
        return bytes(out)

    with (root / 'experts.bin').open('wb') as f:
        for _ in range(L):
            for _e in range(E):
                f.write(q3k_payload(2 * NFF * (LATENT // 256)))   # gate and up
                f.write(iq4_payload(LATENT * (NFF // 32)))        # down

    lines = ['# strata native experts v6: blob fixture (expert_embd %d, expert_ff %d)' % (LATENT, NFF)]
    for l in range(L):
        lines.append('%d 11 20 %d %d' % (l, l * E * blob, blob))
    (root / 'native_experts.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print('fixture written to %s: %d layer, %d experts, blob %d B, D/U %d B'
          % (root, L, E, blob, (root / 'latent_dense_q8.bin').stat().st_size))
    return 0


if __name__ == '__main__':
    sys.exit(main())
