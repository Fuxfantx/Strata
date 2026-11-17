#!/usr/bin/env python3
"""Byte budget of a GGUF pack, by role and by tensor type.

Reads only the metadata section, so it is instant even on a 60 GB pack.  The point is the number every
bandwidth argument needs: how many bytes one decode token has to read, split into the part that is read
in full (dense weights, the output head) and the part touched only for the experts a token selects.

    python port/tools/gguf-budget.py model.gguf [--top N] [--json]

Roles are inferred from the tensor name, which is all a pack carries:
  attn/linear   the per-layer projections     (read in full)
  ffn dense     a dense feed-forward          (read in full)
  expert        ffn_*_exps                    (only the selected experts)
  ple           per-layer embedding tables    (only the rows a token selects)
  embed/head    output and token embedding    (read in full)
  norm/router   router, norms, biases         (read in full)
"""
from __future__ import annotations

import collections
import json
import struct
import sys

# ggml type -> (block elements, block bytes)
GGML_TYPES = {
    0: ("f32", 1, 4), 1: ("f16", 1, 2), 2: ("q4_0", 32, 18), 3: ("q4_1", 32, 20),
    6: ("q5_0", 32, 22), 7: ("q5_1", 32, 24), 8: ("q8_0", 32, 34), 9: ("q8_1", 32, 36),
    10: ("q2_K", 256, 84), 11: ("q3_K", 256, 110), 12: ("q4_K", 256, 144), 13: ("q5_K", 256, 176),
    14: ("q6_K", 256, 210), 15: ("q8_K", 256, 292),
    16: ("iq2_xxs", 256, 66), 17: ("iq2_xs", 256, 74), 18: ("iq3_xxs", 256, 98), 19: ("iq1_s", 256, 50),
    20: ("iq4_nl", 32, 18), 21: ("iq3_s", 256, 110), 22: ("iq2_s", 256, 82), 23: ("iq4_xs", 256, 136),
    24: ("i8", 1, 1), 25: ("i16", 1, 2), 26: ("i32", 1, 4), 27: ("i64", 1, 8), 28: ("f64", 1, 8),
    29: ("iq1_m", 256, 56), 30: ("bf16", 1, 2), 34: ("tq1_0", 256, 54), 35: ("tq2_0", 256, 66),
    143: ("ptq1_0", 128, 26), 144: ("tq2_t", 128, 34),
    145: ("tqk6", 128, 26), 146: ("tqk7", 128, 30),
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
            if t in SCALAR:
                f.read(SCALAR[t])
            elif t == 8:
                rd_str()
            elif t == 9:
                elem = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                if elem in SCALAR:
                    f.read(SCALAR[elem] * n)
                elif elem == 8:
                    for _ in range(n):
                        rd_str()
                else:
                    raise ValueError(f"nested array elem {elem}")
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
            elif vt == 5:
                meta[key] = struct.unpack("<i", f.read(4))[0]
            elif vt == 11:
                meta[key] = struct.unpack("<q", f.read(8))[0]
            elif vt == 6:
                meta[key] = struct.unpack("<f", f.read(4))[0]
            else:
                skip(vt)

        tensors = []
        for _ in range(n_tensors):
            name = rd_str()
            nd = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            ttype = struct.unpack("<I", f.read(4))[0]
            f.read(8)
            tensors.append((name, dims, ttype))
    return meta, tensors


def role_of(name: str) -> str:
    leaf = name.split(".")[-2] if name.count(".") >= 2 else name
    if "_exps" in leaf:
        return "expert"
    if "ple" in name.lower():
        return "ple"
    if leaf in ("token_embd", "output"):
        return "embed/head"
    if leaf.startswith(("ssm_", "attn_", "linear_attn")) or "conv1d" in leaf:
        return "attn/linear"
    if leaf.startswith("ffn_"):
        return "ffn dense"
    if leaf.endswith("norm") or leaf.endswith("_inp") or leaf.endswith("bias") or leaf == "ssm_a":
        return "norm/router"
    return "misc"


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    path = argv[0]
    top = 20
    as_json = "--json" in argv
    if "--top" in argv:
        top = int(argv[argv.index("--top") + 1])

    meta, tensors = read_header(path)
    by_role = collections.Counter()
    by_type = collections.Counter()
    by_role_type = collections.Counter()
    per_layer = collections.Counter()
    biggest = []
    unknown = collections.Counter()

    for name, dims, ttype in tensors:
        if ttype not in GGML_TYPES:
            unknown[ttype] += 1
            continue
        tname, blk, nbytes = GGML_TYPES[ttype]
        nelem = 1
        for d in dims:
            nelem *= d
        size = ((nelem + blk - 1) // blk) * nbytes
        role = role_of(name)
        by_role[role] += size
        by_type[tname] += size
        by_role_type[(role, tname)] += size
        biggest.append((size, name, dims, tname))
        if name.startswith("blk."):
            per_layer[name.split(".")[1]] += size

    total = sum(by_role.values())
    gib = 1024 ** 3

    if as_json:
        print(json.dumps({
            "path": path, "n_tensors": len(tensors), "total_bytes": total,
            "by_role": dict(by_role.most_common()), "by_type": dict(by_type.most_common()),
            "dims": {k: v for k, v in meta.items() if isinstance(v, (int, float, str))},
            "unknown_types": dict(unknown),
        }, indent=2))
        return 0

    print(f"{path}")
    print(f"  tensors {len(tensors)}  known-type total {total / gib:.2f} GiB")
    print("  --- by role ---")
    for k, v in by_role.most_common():
        print(f"  {k:14s} {v / gib:8.3f} GiB  {100.0 * v / total:5.1f}%")
    print("  --- by type ---")
    for k, v in by_type.most_common():
        print(f"  {k:10s} {v / gib:8.3f} GiB  {100.0 * v / total:5.1f}%")
    print("  --- role x type (top 16) ---")
    for (r, t), v in by_role_type.most_common(16):
        print(f"  {r:14s} {t:10s} {v / gib:8.3f} GiB")
    if per_layer:
        vals = sorted(per_layer.values())
        print(f"  --- {len(per_layer)} layers: {vals[0] / 1048576:.1f} .. {vals[-1] / 1048576:.1f} MiB each")
    print(f"  --- top {top} tensors ---")
    for size, name, dims, tname in sorted(biggest, reverse=True)[:top]:
        print(f"  {size / 1048576:9.2f} MiB  {name:42s} {dims} {tname}")
    if unknown:
        print(f"  !! unknown type ids {dict(unknown)} (excluded from the totals)")
    print("  --- metadata ---")
    for k, v in sorted(meta.items()):
        if not k.startswith(("tokenizer", "general.tags")):
            print(f"  {k} = {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
