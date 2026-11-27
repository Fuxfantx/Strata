#!/usr/bin/env python3
"""What would the per-layer embedding table cost at a lower precision, and what would it lose?

The released packs store `per_layer_token_embd.weight` as FP8 (one byte per value, 160 per row).  A row is read whole
and multiplied into the residual stream, so the question is not "how many bits per weight" in the abstract but "what is
the error of the vector we actually use", and what that costs in bytes.

This tool samples rows from the table **as downloaded** (a partial file is fine: rows are independent and the sample is
restricted to the byte range that exists), and reports, per scheme:

    bytes/row     what the row would cost if stored that way
    relative L2   ||decode(stored) - row|| / ||row||
    cosine        1 - cos(decode(stored), row)

Schemes:
    fp8           the file as it is (baseline; error 0 by definition)
    q8_0          ggml Q8_0: 32-value blocks, one f16 scale each
    q4_0          ggml Q4_0: 32-value blocks, one f16 scale each, 16 four-bit codes
    affine4_row   one f16 scale and one f16 zero per row, 4-bit codes
    kmeans4_row   16 centroids fitted to the row itself, 4-bit codes; the centroids travel with the row
    gsq4_32       group-wise scalar quantisation: per 32-value group a f16 scale, 4-bit codes (no zero point)

The last three are the "K-means / GSQ-RCO" family the task asks about; they are here to be measured against Q4_0, not to
be assumed better.

    python port/tools/ple-quant-eval.py --pack DIR [--rows 256] [--seed 1]
"""
import argparse
import json
import pathlib
import struct
import sys

import numpy as np


def fp8_e4m3_to_f32(b: np.ndarray) -> np.ndarray:
    """Decode E4M3 (bias 7, no infinities, 0x7F/0xFF are NaN in this variant)."""
    b = b.astype(np.uint8)
    sign = np.where((b >> 7) != 0, -1.0, 1.0)
    exp = ((b >> 3) & 0x0F).astype(np.int32)
    man = (b & 0x07).astype(np.float64)
    out = np.where(exp == 0, man / 8.0 * 2.0 ** -6, (1.0 + man / 8.0) * 2.0 ** (exp - 7))
    out = sign * out
    out[(b & 0x7F) == 0x7F] = np.nan
    return out.astype(np.float32)


def fp8_e5m2_to_f32(b: np.ndarray) -> np.ndarray:
    b = b.astype(np.uint8)
    sign = np.where((b >> 7) != 0, -1.0, 1.0)
    exp = ((b >> 2) & 0x1F).astype(np.int32)
    man = (b & 0x03).astype(np.float64)
    out = np.where(exp == 0, man / 4.0 * 2.0 ** -14, (1.0 + man / 4.0) * 2.0 ** (exp - 15))
    return (sign * out).astype(np.float32)


def read_gguf_header(path, want):
    """(name, type, offset in file, dims) for the tensor called `want`, from a GGUF header."""
    with open(path, 'rb') as f:
        head = f.read(24)
        if head[:4] != b'GGUF':
            raise SystemExit('%s: not a GGUF file' % path)
        n_tensors = struct.unpack_from('<Q', head, 8)[0]
        n_kv = struct.unpack_from('<Q', head, 16)[0]

        def rstr():
            n = struct.unpack('<Q', f.read(8))[0]
            return f.read(n).decode('utf-8', 'replace')

        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        for _ in range(n_kv):
            rstr()
            t = struct.unpack('<I', f.read(4))[0]
            if t == 8:
                rstr()
            elif t == 9:
                at = struct.unpack('<I', f.read(4))[0]
                n = struct.unpack('<Q', f.read(8))[0]
                for _i in range(n):
                    if at == 8:
                        rstr()
                    else:
                        f.read(sizes[at])
            else:
                f.read(sizes[t])
        found = None
        for _ in range(n_tensors):
            name = rstr()
            nd = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
            ttype = struct.unpack('<I', f.read(4))[0]
            off = struct.unpack('<Q', f.read(8))[0]
            if name == want:
                found = (name, ttype, off, dims)
        data_start = (f.tell() + 31) // 32 * 32
    if found is None:
        raise SystemExit('%s: no tensor %s' % (path, want))
    name, ttype, off, dims = found
    return name, ttype, data_start + off, dims


def q8_0_roundtrip(rows: np.ndarray) -> np.ndarray:
    """ggml Q8_0: per 32 values one f16 scale, int8 codes."""
    n, k = rows.shape
    out = np.empty_like(rows)
    for i in range(n):
        for b in range(k // 32):
            seg = rows[i, b * 32:(b + 1) * 32]
            d = np.float16(np.abs(seg).max() / 127.0 or 1e-8)
            q = np.clip(np.round(seg / float(d)), -127, 127).astype(np.int8)
            out[i, b * 32:(b + 1) * 32] = d * q
    return out


def q4_0_roundtrip(rows: np.ndarray) -> np.ndarray:
    """ggml Q4_0: per 32 values one f16 scale, codes 0..15 centred on 8."""
    n, k = rows.shape
    out = np.empty_like(rows)
    for i in range(n):
        for b in range(k // 32):
            seg = rows[i, b * 32:(b + 1) * 32]
            d = np.float16(np.abs(seg).max() / 7.0 or 1e-8)
            q = np.clip(np.round(seg / float(d)) + 8, 0, 15)
            out[i, b * 32:(b + 1) * 32] = (q - 8) * float(d)
    return out


def affine4_roundtrip(rows: np.ndarray) -> np.ndarray:
    """One zero and one step per row, 4-bit codes: 2 + 2 + k/2 bytes."""
    n, k = rows.shape
    out = np.empty_like(rows)
    for i in range(n):
        lo, hi = float(rows[i].min()), float(rows[i].max())
        step = (hi - lo) / 15.0 if hi > lo else 1e-8
        q = np.clip(np.round((rows[i] - lo) / step), 0, 15)
        out[i] = lo + q * step
    return out


def gsq4_roundtrip(rows: np.ndarray) -> np.ndarray:
    """Group-wise scalar quantisation: one scale per 32 values, 4-bit codes, no zero point."""
    n, k = rows.shape
    out = np.empty_like(rows)
    for i in range(n):
        for b in range(k // 32):
            seg = rows[i, b * 32:(b + 1) * 32]
            d = np.abs(seg).max() / 7.0
            if d == 0:
                d = 1e-8
            q = np.clip(np.round(seg / d), -8, 7)
            out[i, b * 32:(b + 1) * 32] = q * d
    return out


def kmeans4_roundtrip(rows: np.ndarray, iters: int = 12) -> np.ndarray:
    """16 centroids fitted to each row (Lloyd), 4-bit codes; the centroids travel with the row."""
    n, k = rows.shape
    out = np.empty_like(rows)
    for i in range(n):
        v = rows[i]
        qs = np.quantile(v, np.linspace(0, 1, 16))
        c = qs.astype(np.float32)
        for _ in range(iters):
            assign = np.abs(v[:, None] - c[None, :]).argmin(axis=1)
            for j in range(16):
                m = assign == j
                if m.any():
                    c[j] = v[m].mean()
        assign = np.abs(v[:, None] - c[None, :]).argmin(axis=1)
        out[i] = c[assign]
    return out


def rel_l2(got, ref):
    return float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-12))


def cosine_err(got, ref):
    num = np.sum(got * ref, axis=1)
    den = np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1)
    return float(np.mean(1.0 - num / np.maximum(den, 1e-12)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--pack', required=True, help='the model folder that holds ple-fp8.gguf')
    ap.add_argument('--file', default='ple-fp8.gguf')
    ap.add_argument('--tensor', default='per_layer_token_embd.weight')
    ap.add_argument('--rows', type=int, default=256)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    path = pathlib.Path(args.pack) / args.file
    if not path.exists():
        part = path.with_suffix(path.suffix + '.part')
        if part.exists():
            path = part
            print('note: reading the partial file', path.name)
        else:
            raise SystemExit('%s is not there' % path)

    name, ttype, data_off, dims = read_gguf_header(path, args.tensor)
    k, n_rows = int(dims[0]), int(dims[1])
    size = path.stat().st_size
    available_rows = max(0, (size - data_off) // k)
    if available_rows <= 0:
        raise SystemExit('no row of %s is complete in the %d bytes that exist' % (name, size))
    print('%s: %s, %d rows of %d values, %.2f GiB in the file now, %d rows readable'
          % (path.name, ggml_type_name(ttype), n_rows, k, size / 2 ** 30, available_rows))

    rng = np.random.default_rng(args.seed)
    n = min(args.rows, available_rows)
    pick = np.sort(rng.choice(available_rows, size=n, replace=False))

    if ttype == 24:       # GGML_TYPE_I8: the table in the released packs is one signed byte per value
        decode = lambda b: b.view(np.int8).astype(np.float32)     # noqa: E731 - a one-line decode is the honest one
    elif ttype == 36:
        decode = fp8_e4m3_to_f32
    elif ttype == 37:
        decode = fp8_e5m2_to_f32
    else:
        raise SystemExit('this tool handles int8 and fp8 tables; the file holds type %d (%s)'
                         % (ttype, ggml_type_name(ttype)))

    ref = np.empty((n, k), dtype=np.float32)
    with open(path, 'rb') as f:
        for i, r in enumerate(pick):
            f.seek(data_off + int(r) * k)
            ref[i] = decode(np.frombuffer(f.read(k), dtype=np.uint8))

    schemes = [
        ('fp8',          k,                 None),
        ('q8_0',         k + 2 * (k // 32), q8_0_roundtrip),
        ('q4_0',         (k // 2) + 2 * (k // 32), q4_0_roundtrip),
        ('gsq4_32',      (k // 2) + 2 * (k // 32), gsq4_roundtrip),
        ('affine4_row',  4 + k // 2,        affine4_roundtrip),
        ('kmeans4_row',  32 + k // 2,       kmeans4_roundtrip),
    ]
    print('%-13s %10s %10s %10s %10s' % ('scheme', 'bytes/row', '% of fp8', 'rel L2', '1-cos'))
    base_bytes = k
    for label, nbytes, fn in schemes:
        if fn is None:
            print('%-13s %10d %10.1f %10s %10s' % (label, nbytes, 100.0 * nbytes / base_bytes, '-', '-'))
            continue
        got = fn(ref)
        print('%-13s %10d %10.1f %10.4f %10.5f'
              % (label, nbytes, 100.0 * nbytes / base_bytes, rel_l2(got, ref), cosine_err(got, ref)))
    print('\n%d rows sampled from the %d that exist; error is measured against the FP8 decode of the same rows, so it '
          'does not include the FP8 error the file already has.' % (n, available_rows))
    return 0


# The type ids as the library reports them (ggml_type_name / ggml_blck_size on the built ggml-base.dll).  They are not
# upstream's numbers and cannot be counted out of the enum header: that header has removed entries and explicit values.
GGML_TYPE_NAMES = {
    0: 'f32', 1: 'f16', 2: 'q4_0', 3: 'q4_1', 6: 'q5_0', 7: 'q5_1', 8: 'q8_0', 9: 'q8_1',
    10: 'q2_K', 11: 'q3_K', 12: 'q4_K', 13: 'q5_K', 14: 'q6_K', 15: 'q8_K',
    16: 'iq2_xxs', 17: 'iq2_xs', 18: 'iq3_xxs', 19: 'iq1_s', 20: 'iq4_nl', 21: 'iq3_s', 22: 'iq2_s', 23: 'iq4_xs',
    24: 'i8', 25: 'i16', 26: 'i32', 27: 'i64', 28: 'f64', 29: 'iq1_m', 30: 'bf16',
    34: 'tq1_0', 35: 'tq2_0', 39: 'mxfp4',
}


def ggml_type_name(t):
    return GGML_TYPE_NAMES.get(t, 'type %d' % t)


if __name__ == '__main__':
    sys.exit(main())
