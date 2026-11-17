#!/usr/bin/env python3
"""What a trellis -> int8 migration of the expert weights would cost and buy, on this machine.

The trellis expert matvec is the one bottleneck left on both backends (port/docs/PERFORMANCE.md sections 11 and 14): 0.47 GiB per
token at 8.8 GB/s on the CPU and 10 GB/s on the GPU, against a 137 / 108 GB/s wall.  Every loop, occupancy, dispatch and layout
lever inside the format has been measured and closed, so the remaining option is to spend bytes to save instructions: expand each
trellis block to int8 at load time and let the kernel use the integer dot product, which is what the q4_0/q8_0 experts of
35B-A3B already do at ~65 GB/s.

    python port/tools/trellis-migration.py --pack Gyro-S-TQ1_0.gguf

The arithmetic is here, in one place, so the numbers in the docs are reproducible.  Measured inputs are marked; everything else is
arithmetic on top of them.
"""
from __future__ import annotations

import argparse
import sys

GIB = 1024 ** 3

# measured on this machine, this session (port/docs/PERFORMANCE.md)
DENSE_BYTES = 3.06 * GIB        # per token, dense + attention + head (gguf-budget.py, then confirmed by the op tables)
DENSE_RATE = 112e9             # GB/s: the CPU dense MUL_MAT after the big-core fix (section 8)
READ_WALL = 137e9              # GB/s, three-channel LPDDR
CPU_TRELLIS_RATE = 8.8e9       # GB/s: the trellis MUL_MAT_ID, measured (section 6.4)
GPU_TRELLIS_RATE = 10e9        # GB/s: the same kernel on the Vulkan backend (section 11)
GPU_DENSE_RATE = 118e9         # GB/s: the GPU's matmuls together (section 6.5)
INT8_ANCHOR_LOW = 65e9         # GB/s: 1.86 GB/token at 35 tok/s on 35B-A3B, whose experts are q4_0: int-dot, whole token
INT8_ANCHOR_HIGH = 110e9       # GB/s: the same class of kernel if it reaches the memory wall like the dense path does
INT8_ANCHOR_MEASURED = 86e9    # GB/s: 35B-A3B on the CPU after this session's changes: 1.86 GB/token at 46.37 tok/s

# q8_0 is the type to migrate to, and the reason matters: it already has an integer-dot kernel on both backends, so the migration
# is a repack and nothing else - no new CPU vec_dot, no new Vulkan shader, no new ggml type.  Its block is 32 weights plus one f16
# scale (34 bytes), i.e. 8.5 bits per weight rather than 8.
Q8_0_BITS_PER_WEIGHT = 8.5

# a trellis weight count is what an int8 expansion pays for: bytes * 8 / bits-per-weight
EXPERT_BITS_PER_WEIGHT = {"tqk6": 1.625, "tqk7": 1.875}


def expert_bytes(pack_gib: float, expert_gib: float, tqk6_share: float) -> tuple[float, float]:
    """(tqk6 GiB, tqk7 GiB) from the pack's expert total and the measured tqk6/tqk7 split (15.234 / 8.789 on Gyro-S)."""
    t6 = expert_gib * tqk6_share
    return t6, expert_gib - t6


def int8_bytes(t6: float, t7: float) -> float:
    """Both trellis types as q8_0 (8.5 bits per weight, the block's f16 scale included)."""
    return (t6 * Q8_0_BITS_PER_WEIGHT / EXPERT_BITS_PER_WEIGHT["tqk6"]
            + t7 * Q8_0_BITS_PER_WEIGHT / EXPERT_BITS_PER_WEIGHT["tqk7"])


def u16_state_bytes(t6: float, t7: float) -> float:
    """The other format option: keep the weights, store the 32 states of a block as uint16 (26 -> 66 and 30 -> 70 bytes)."""
    return t6 * 66 / 26 + t7 * 70 / 30


def decode_ms(per_token_expert_gib: float, rate: float, dense_rate: float) -> float:
    return DENSE_BYTES / dense_rate * 1e3 + per_token_expert_gib * GIB / rate * 1e3


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-gib", type=float, default=54.46, help="the pack as it is (Gyro-S)")
    ap.add_argument("--expert-gib", type=float, default=24.02)
    ap.add_argument("--tqk6-share", type=float, default=15.234 / 24.02)
    ap.add_argument("--top", type=int, default=10, help="experts a token selects")
    ap.add_argument("--n-expert", type=int, default=512)
    ap.add_argument("--ram-gib", type=float, default=47.6)
    a = ap.parse_args(argv)

    t6, t7 = expert_bytes(a.pack_gib, a.expert_gib, a.tqk6_share)
    keep = a.top / a.n_expert                       # the fraction of the expert bytes a token reads
    now_pt = a.expert_gib * keep
    i8_total, i8_pt = int8_bytes(t6, t7), int8_bytes(t6, t7) * keep
    u16_total, u16_pt = u16_state_bytes(t6, t7), u16_state_bytes(t6, t7) * keep

    print(f"pack {a.pack_gib:.2f} GiB, experts {a.expert_gib:.2f} (tqk6 {t6:.2f} @1.625 bpw, tqk7 {t7:.2f} @1.875 bpw)")
    print(f"a token reads {keep:.4f} of the experts: {now_pt:.3f} GiB now\n")
    print(f"{'format':28s} {'experts':>10s} {'pack':>9s} {'/token':>8s}  {'CPU decode':>10s}  {'GPU decode':>10s}")
    rows = [
        ("trellis (today)", a.expert_gib, a.pack_gib, now_pt, CPU_TRELLIS_RATE, GPU_TRELLIS_RATE),
        ("states as uint16", u16_total, a.pack_gib - a.expert_gib + u16_total, u16_pt, CPU_TRELLIS_RATE * 1.4, GPU_TRELLIS_RATE * 1.4),
        (f"q8_0, {INT8_ANCHOR_LOW / 1e9:.0f} GB/s", i8_total, a.pack_gib - a.expert_gib + i8_total, i8_pt, INT8_ANCHOR_LOW, INT8_ANCHOR_LOW),
        (f"q8_0, {INT8_ANCHOR_MEASURED / 1e9:.0f} GB/s (measured)", i8_total, a.pack_gib - a.expert_gib + i8_total, i8_pt,
         INT8_ANCHOR_MEASURED, INT8_ANCHOR_MEASURED),
        (f"q8_0, {INT8_ANCHOR_HIGH / 1e9:.0f} GB/s", i8_total, a.pack_gib - a.expert_gib + i8_total, i8_pt, INT8_ANCHOR_HIGH, INT8_ANCHOR_HIGH),
    ]
    for name, exp, pack, pt, cpu_rate, gpu_rate in rows:
        cpu = 1000.0 / decode_ms(pt, cpu_rate, DENSE_RATE)
        gpu = 1000.0 / decode_ms(pt, gpu_rate, GPU_DENSE_RATE)
        print(f"{name:28s} {exp:9.1f}G {pack:8.0f}G {pt:7.3f}G  {cpu:9.1f} t/s  {gpu:9.1f} t/s")
    print()
    print("the CPU column is the token's matmul time and is what the measured 11.8-12.1 tok/s reproduces; the GPU column counts the")
    print("matmuls only and ignores the ~60 ms of small-op time a GPU token also spends, so it is an upper bound (it says 12.8 for")
    print("a measured 10.0).  Read the two columns as 'what the expert kernel would allow', not as a forecast.")
    print(f"the machine has {a.ram_gib:.1f} GiB: the q8_0 pack is {a.pack_gib - a.expert_gib + i8_total:.0f} GiB, i.e. mapped, "
          f"never resident - the same mode the 54 GiB pack is in today")
    print(f"a token then moves {DENSE_BYTES / GIB + i8_pt:.2f} GiB against {DENSE_BYTES / GIB + now_pt:.2f} GiB now "
          f"({(DENSE_BYTES / GIB + i8_pt) / (DENSE_BYTES / GIB + now_pt):.2f}x), so the win needs the kernel to be more than "
          f"that much faster")
    print("\npartial expansion buys nothing: the expert weights are read by layer and by expert, so K of 48 layers expands to")
    for k in (4, 8, 12, 24, 48):
        print(f"  {k:2d} layers: {i8_total * k / 48:6.1f} GiB of int8 experts, "
              f"{100.0 * k / 48 * (1 - CPU_TRELLIS_RATE / INT8_ANCHOR_LOW):5.1f}% off the CPU expert time at the "
              f"{INT8_ANCHOR_LOW / 1e9:.0f} GB/s anchor")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
