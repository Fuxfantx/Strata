"""Minimal GGUF header/tensor-info reader: the dims behind the GDN fusion (no external packages here)."""
import struct
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Fuxfantx\C-Res\Qwen\3.6-35A3\Qwen3.6-35B-A3B-Q4_0.gguf"

# value type sizes for the scalar/metadata types (array handled separately)
FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")

def skip_value(f, t):
    if t in FIXED:
        f.read(FIXED[t])
    elif t == 8:
        read_str(f)
    elif t == 9:
        at = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if at in FIXED:
            f.read(FIXED[at] * n)
        elif at == 8:
            for _ in range(n):
                read_str(f)
        else:
            raise ValueError(f"array elem type {at}")
    else:
        raise ValueError(f"type {t}")

def read_value(f, t):
    if t == 8:
        return read_str(f)
    if t == 4:
        return struct.unpack("<I", f.read(4))[0]
    if t == 5:
        return struct.unpack("<i", f.read(4))[0]
    if t == 6:
        return struct.unpack("<f", f.read(4))[0]
    if t == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if t == 11:
        return struct.unpack("<q", f.read(8))[0]
    if t == 12:
        return struct.unpack("<d", f.read(8))[0]
    if t == 7:
        return bool(f.read(1)[0])
    skip_value(f, t)
    return None

with open(PATH, "rb") as f:
    assert f.read(4) == b"GGUF", "not a GGUF"
    version = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]

    meta = {}
    for _ in range(n_kv):
        key = read_str(f)
        t = struct.unpack("<I", f.read(4))[0]
        if t == 9:                      # arrays: skip, we only want scalars
            skip_value(f, t)
            continue
        meta[key] = read_value(f, t)

    print(f"version {version}, {n_tensors} tensors, {n_kv} kv")
    for k in sorted(meta):
        if any(s in k for s in ("ssm", "n_embd", "n_head", "arch", "expert", "n_ff", "linear")):
            print(f"  {k} = {meta[k]}")

    want = ("ssm_conv1d", "ssm_a", "ssm_dt", "ssm_norm", "ssm_out", "attn_qkv", "attn_gate",
            "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate_inp", "ffn_gate_exps", "ffn_down_exps",
            "ffn_up_exps", "ffn_gate_shexp", "ffn_down_shexp", "ffn_up_shexp", "token_embd", "output_norm")
    seen = {}
    for _ in range(n_tensors):
        name = read_str(f)
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
        ttype = struct.unpack("<I", f.read(4))[0]
        struct.unpack("<Q", f.read(8))[0]
        short = name.split(".")[-2] + "." + name.split(".")[-1] if name.count(".") >= 2 else name
        if any(w == name.split(".")[-2] or w == name.split(".")[-1] for w in want):
            layer0 = name.startswith("blk.0.") or name in ("token_embd.weight", "output_norm.weight")
            if layer0 and short not in seen:
                seen[short] = (name, dims, ttype)
    print("  --- blk.0 tensors ---")
    for short in sorted(seen):
        name, dims, ttype = seen[short]
        print(f"  {name:34s} dims={dims} type={ttype}")
