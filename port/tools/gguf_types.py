#!/usr/bin/env python3
"""Print the tensor-type histogram and the key hyperparameters of a GGUF file.

Reads only the metadata section, so it is instant even on a 60 GB pack.  The histogram is what decides which kernels a
model runs: an int-dot-MMQ quant (q4_0/q4_K/q5_0/q8_0) takes the fast mat-mat path, a trellis type (TQ2_T/PTQ1_0) does
not exist there at all.

    python port/tools/gguf_types.py model.gguf [more.gguf ...]
"""
from __future__ import annotations

import collections
import struct
import sys

TYPES = {
    0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0", 9: "q8_1",
    41: "q1_0", 42: "q2_0",
    10: "q2_K", 11: "q3_K", 12: "q4_K", 13: "q5_K", 14: "q6_K", 15: "q8_K",
    16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs", 19: "iq1_s", 20: "iq4_nl", 21: "iq3_s",
    22: "iq2_s", 23: "iq4_xs", 24: "i8", 25: "i16", 26: "i32", 27: "i64", 28: "f64",
    29: "iq1_m", 30: "bf16", 34: "tq1_0", 35: "tq2_0",
    143: "ptq1_0", 144: "tq2_t",
}
SCALAR = {0: 1, 1: 1, 7: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 10: 8, 11: 8, 12: 8}


def read_header(path: str):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = struct.unpack("<I", f.read(4))[0]
        if version < 2:
            raise ValueError(f"unsupported GGUF version {version}")
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def rd_str() -> str:
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")

        def skip(t: int) -> None:
            if t in (0, 1, 7):
                f.read(1)
            elif t in (2, 3):
                f.read(2)
            elif t in (4, 5, 6):
                f.read(4)
            elif t == 8:
                rd_str()
            elif t == 9:
                elem = struct.unpack("<I", f.read(4))[0]
                for _ in range(struct.unpack("<Q", f.read(8))[0]):
                    skip(elem)
            elif t in (10, 11, 12):
                f.read(8)
            else:
                raise ValueError(f"bad kv type {t}")

        meta = {}
        for _ in range(n_kv):
            key = rd_str()
            vt = struct.unpack("<I", f.read(4))[0]
            if vt == 8:
                meta[key] = rd_str()
            elif vt == 4:
                meta[key] = struct.unpack("<I", f.read(4))[0]
            elif vt == 10:
                meta[key] = struct.unpack("<Q", f.read(8))[0]
            else:
                skip(vt)

        counts = collections.Counter()
        for _ in range(n_tensors):
            rd_str()
            nd = struct.unpack("<I", f.read(4))[0]
            for _ in range(nd):
                f.read(8)
            counts[TYPES.get(struct.unpack("<I", f.read(4))[0], "?")] += 1
            f.read(8)
    return meta, counts


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for path in argv:
        try:
            meta, counts = read_header(path)
        except Exception as e:  # a truncated or partial file is the common case here
            print(f"{path}: {e}")
            continue
        arch = meta.get("general.architecture", "?")
        name = meta.get("general.name", "")
        total = sum(counts.values())
        hist = ", ".join(f"{k} x{v}" for k, v in counts.most_common())
        print(f"{path}")
        print(f"  arch {arch}  name {name}  file_type {meta.get('general.file_type', '?')}  tensors {total}")
        print(f"  types: {hist}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
