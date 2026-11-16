"""List the tensor names of one layer, to find how a pack names its expert stacks."""
import struct
import sys

PATH = sys.argv[1]
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "blk.0."
FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

with open(PATH, "rb") as f:
    assert f.read(4) == b"GGUF"
    struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]

    def rs():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")

    for _ in range(n_kv):
        rs()
        t = struct.unpack("<I", f.read(4))[0]
        if t in FIXED:
            f.read(FIXED[t])
        elif t == 8:
            rs()
        elif t == 9:
            at = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            if at in FIXED:
                f.read(FIXED[at] * n)
            else:
                for _ in range(n):
                    rs()

    out = []
    for _ in range(n_tensors):
        name = rs()
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
        struct.unpack("<I", f.read(4))[0]
        struct.unpack("<Q", f.read(8))[0]
        if name.startswith(PREFIX):
            out.append((name, dims))
    print(f"{PREFIX}* tensors: {len(out)}")
    for name, dims in sorted(out):
        print(f"  {name:44s} {dims}")
