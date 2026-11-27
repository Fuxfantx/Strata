#!/usr/bin/env python3
"""A float32 reference for the latent block, and a tiny fixture to test against.

The block is, per layer and per token: `xl = D[group] @ x`, the routed expert runs `xl -> gate/up -> SwiGLU -> down`,
then `y = U[group] @ h`, weighted and summed over the routed slots.  This file implements that arithmetic in numpy from
the pack's own byte layout, so the Vulkan path has something independent to be compared with.

Scope, stated plainly: the D/U rows (Q8_0, Q4_0) are dequantised here, because those formats are simple and their
dequantisation is exactly defined.  The expert rows (Q3_K, IQ4_NL) are **not** dequantised here - the authoritative
dequantiser for them is ggml's own, and the comparison for that stage is made on the C++ side, where ggml's code can be
called.  `--fixture` writes a tiny pack whose expert stage is float32, so the block's shape and combination logic can be
checked before the quantised stage exists.

    python port/tools/latent-ref.py --self-test
    python port/tools/latent-ref.py --fixture OUTDIR
"""
import argparse
import json
import pathlib
import struct
import sys

import numpy as np

QK = 32
Q8_0 = 8
Q4_0 = 2


def deq_q8_0(buf, n):
    """Q8_0: per 32 values {f16 d; 32 int8 q}; value = d * q."""
    row = np.frombuffer(buf, dtype=np.uint8)
    blocks = row.size // 34
    out = np.empty(blocks * 32, dtype=np.float32)
    for b in range(blocks):
        base = b * 34
        d = np.frombuffer(row[base:base + 2].tobytes(), dtype=np.float16)[0].astype(np.float32)
        q = np.frombuffer(row[base + 2:base + 34].tobytes(), dtype=np.int8).astype(np.float32)
        out[b * 32:(b + 1) * 32] = d * q
    return out[:n]


def deq_q4_0(buf, n):
    """Q4_0: per 32 values {f16 d; 16 bytes, low nibble then high}; value = (code - 8) * d."""
    row = np.frombuffer(buf, dtype=np.uint8)
    blocks = row.size // 18
    out = np.empty(blocks * 32, dtype=np.float32)
    for b in range(blocks):
        base = b * 18
        d = np.frombuffer(row[base:base + 2].tobytes(), dtype=np.float16)[0].astype(np.float32)
        qs = row[base + 2:base + 18]
        lo = (qs & 0x0F).astype(np.int8) - 8
        hi = (qs >> 4).astype(np.int8) - 8
        out[b * 32:(b + 1) * 32] = d * np.concatenate([lo, hi]).astype(np.float32)
    return out[:n]


def deq_f32(buf, n):
    return np.frombuffer(buf, dtype=np.float32)[:n].astype(np.float32)


def deq_f16(buf, n):
    return np.frombuffer(buf, dtype=np.float16)[:n].astype(np.float32)


IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)


def deq_iq4_nl(buf, n):
    """IQ4_NL: 32 values per 18-byte block {half d; uint8 qs[16]}, values from the format's own table."""
    row = np.frombuffer(buf, dtype=np.uint8)
    blocks = row.size // 18
    out = np.empty(blocks * 32, dtype=np.float32)
    for b in range(blocks):
        base = b * 18
        d = np.frombuffer(row[base:base + 2].tobytes(), dtype=np.float16)[0].astype(np.float32)
        qs = row[base + 2:base + 18]
        lo = qs & 0x0F
        hi = qs >> 4
        out[b * 32:(b + 1) * 32] = d * IQ4NL[np.concatenate([lo, hi])]
    return out[:n]


def deq_q3_K(buf, n):
    """Q3_K, decoded from the format definition and checked against ggml by port/tools/q3k-check.py."""
    row = np.frombuffer(buf, dtype=np.uint8)
    blocks = row.size // 110
    out = np.empty(blocks * 256, dtype=np.float32)
    for b in range(blocks):
        blk = row[b * 110:(b + 1) * 110]
        hmask = blk[0:32]
        qs = blk[32:96]
        d_all = np.frombuffer(blk[108:110].tobytes(), dtype=np.float16)[0].astype(np.float32)
        a = np.frombuffer(blk[96:108].tobytes(), dtype=np.uint32).astype(np.uint64)
        tmp = int(a[2])
        km1, km2 = 0x03030303, 0x0F0F0F0F
        words = [
            int((a[0] & km2) | (((tmp >> 0) & km1) << 4)),
            int((a[1] & km2) | (((tmp >> 2) & km1) << 4)),
            int(((a[0] >> 4) & km2) | (((tmp >> 4) & km1) << 4)),
            int(((a[1] >> 4) & km2) | (((tmp >> 6) & km1) << 4)),
        ]
        scales = []
        for w in words:
            scales.extend([((w >> (8 * i)) & 0xFF) - 32 for i in range(4)])
        vals = np.zeros(256, dtype=np.float32)
        is_ = 0
        m = 1
        qi = 0
        for nh in range(0, 256, 128):
            shift = 0
            for j in range(4):
                for l in range(16):
                    q2 = int((qs[qi + l] >> shift) & 3)
                    bit = 0 if (int(hmask[l]) & m) else 4
                    vals[nh + j * 32 + l] = d_all * scales[is_] * (q2 - bit)
                is_ += 1
                for l in range(16):
                    q2 = int((qs[qi + l + 16] >> shift) & 3)
                    bit = 0 if (int(hmask[l + 16]) & m) else 4
                    vals[nh + j * 32 + 16 + l] = d_all * scales[is_] * (q2 - bit)
                is_ += 1
                shift += 2
                m <<= 1
            qi += 32
        out[b * 256:(b + 1) * 256] = vals
    return out[:n]


DEQ = {Q8_0: deq_q8_0, Q4_0: deq_q4_0, 0: deq_f32, 1: deq_f16, 11: deq_q3_K, 20: deq_iq4_nl}
# bytes and values per block, per type: Q3_K packs 256 values into 110 bytes, the others 32 into 18 or 34
TYPE_BLOCK = {Q8_0: (32, 34), Q4_0: (32, 18), 11: (256, 110), 20: (32, 18), 0: (1, 4), 1: (1, 2)}


def deq_rows(buf, ggml_type, rows, n_values):
    """Dequantise `rows` consecutive rows of `n_values` values each."""
    if ggml_type not in DEQ:
        raise SystemExit('type %d is not dequantised here (see the module docstring)' % ggml_type)
    if ggml_type in TYPE_BLOCK:
        blk, blk_bytes = TYPE_BLOCK[ggml_type]
        nb = blk_bytes * (n_values // blk)
    elif ggml_type == 0:
        nb = n_values * 4
    elif ggml_type == 1:
        nb = n_values * 2
    else:
        raise SystemExit('type %d has no row size here' % ggml_type)
    out = np.empty((rows, n_values), dtype=np.float32)
    for r in range(rows):
        out[r] = DEQ[ggml_type](buf[r * nb:(r + 1) * nb], n_values)
    return out


def gguf_tensors(path):
    """{name: (dims, type, offset_into_data)} plus the absolute data start of a GGUF file."""
    with open(path, 'rb') as f:
        head = f.read(24)
        if head[:4] != b'GGUF':
            raise SystemExit('%s: not a GGUF file' % path)
        n_tensors = struct.unpack_from('<Q', head, 8)[0]
        n_kv = struct.unpack_from('<Q', head, 16)[0]

        def rstr():
            n = struct.unpack('<Q', f.read(8))[0]
            return f.read(n).decode('utf-8', 'replace')

        for _ in range(n_kv):
            rstr()
            t = struct.unpack('<I', f.read(4))[0]
            if t == 8:
                rstr()
            elif t == 9:
                at = struct.unpack('<I', f.read(4))[0]
                n = struct.unpack('<Q', f.read(8))[0]
                sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
                for _i in range(n):
                    if at == 8:
                        rstr()
                    else:
                        f.read(sizes[at])
            else:
                f.read({0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[t])
        out = {}
        for _ in range(n_tensors):
            name = rstr()
            nd = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
            ttype = struct.unpack('<I', f.read(4))[0]
            off = struct.unpack('<Q', f.read(8))[0]
            out[name] = (dims, ttype, off)
        data_start = (f.tell() + 31) // 32 * 32
    return out, data_start


def gguf_tensor_bytes(path, name, table=None, data_start=None):
    if table is None:
        table, data_start = gguf_tensors(path)
    dims, ttype, off = table[name]
    n = tensor_bytes(ttype, dims)
    with open(path, 'rb') as f:
        f.seek(data_start + off)
        return f.read(n)


def tensor_bytes(ggml_type, dims):
    sizes = {0: (1, 4), 1: (1, 2), 2: (32, 18), 8: (32, 34), 11: (256, 110), 20: (32, 18), 26: (1, 4)}
    block, size = sizes[ggml_type]
    n = 1
    for d in dims:
        n *= d
    return n // block * size


def block(x, xl, gate_rows, up_rows, down_rows, u_rows, y_weights):
    """The latent block's arithmetic in float32, given already-dequantised pieces.

    x            [H]          the token's width-H activation
    xl           [G, ELL]     the D projection of x, one row per routing group
    gate_rows    list of 10 -> [n_ff] gate(xl[g])
    up_rows      list of 10 -> [n_ff] up(xl[g])
    down_rows    list of 10 -> [ELL] down(silu(gate)*up)
    y_weights    [10]         routing weights, already normalised
    """
    out = np.zeros_like(x)
    for s in range(len(gate_rows)):
        h = gate_rows[s] / (1.0 + np.exp(-gate_rows[s])) * up_rows[s]
        y = down_rows[s]                       # [ELL]: the expert's down projection
        out += y_weights[s] * (u_rows[s] @ y)  # [H]: the group's U projection
    return out


def self_test() -> int:
    rng = np.random.default_rng(7)
    H, ELL, G = 64, 48, 4

    # A D matrix whose rows are exactly representable in Q8_0 (integers scaled by 1/8) so the dequantiser can be
    # checked against the values that were quantised, not against a tolerance.
    n_rows = G * ELL
    d_f32 = rng.integers(-8, 8, size=(n_rows, H)).astype(np.float32) / 8.0
    buf = bytearray()
    scale = 1.0 / 8.0
    for r in range(n_rows):
        q = np.round(d_f32[r] / scale).astype(np.int8)
        for blk in range(H // QK):          # one scale per 32 values, that is the Q8_0 block
            buf += struct.pack('<e', scale)
            buf += q[blk * QK:(blk + 1) * QK].tobytes()
    got = deq_rows(bytes(buf), Q8_0, n_rows, H)
    err = np.abs(got - d_f32).max()
    print('Q8_0 round trip: max abs error %.6g' % err)

    # the same for Q4_0: values are multiples of d, codes 0..15
    d_f32 = (rng.integers(-8, 8, size=(n_rows, H)) * 2).astype(np.float32)   # multiples of the Q4_0 scale
    buf = bytearray()
    scale = 2.0
    for r in range(n_rows):
        code = np.clip(np.round(d_f32[r] / scale) + 8, 0, 15).astype(np.uint8)
        for blk in range(H // QK):          # one block of 32 values: {f16 d; 16 bytes, low nibble then high}
            c = code[blk * QK:(blk + 1) * QK]
            qs = ((c[:16] & 0x0F) | ((c[16:] & 0x0F) << 4)).astype(np.uint8)
            buf += struct.pack('<e', scale)
            buf += qs.tobytes()
    got = deq_rows(bytes(buf), Q4_0, n_rows, H)
    err4 = np.abs(got - d_f32).max()
    print('Q4_0 round trip: max abs error %.6g' % err4)

    # the block's combination: two slots, one group each, hand-checked
    x = np.zeros(H, dtype=np.float32)
    xl = rng.standard_normal((G, ELL)).astype(np.float32)
    gate = [rng.standard_normal(32).astype(np.float32) for _ in range(2)]
    up = [rng.standard_normal(32).astype(np.float32) for _ in range(2)]
    down = [rng.standard_normal(ELL).astype(np.float32) for _ in range(2)]
    w = np.array([0.25, 0.75], dtype=np.float32)
    u_rows = [rng.standard_normal((H, ELL)).astype(np.float32) for _ in range(2)]
    out = block(x, xl, gate, up, down, u_rows, w)
    manual = np.zeros(H, dtype=np.float32)
    for s in range(2):
        h = gate[s] / (1.0 + np.exp(-gate[s])) * up[s]
        manual += w[s] * (u_rows[s] @ down[s])
    errb = np.abs(out - manual).max()
    print('block combination: max abs error %.6g' % errb)

    ok = err < 1e-6 and err4 < 1e-6 and errb < 1e-6
    print('result:', 'ok' if ok else 'FAILED')
    return 0 if ok else 1


def fixture(outdir: pathlib.Path) -> int:
    """A tiny latent pack: 2 layers, 4 groups, ELL 16, n_ff 8, 8 experts, f32 experts."""
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    H, ELL, G, FF, E, L = 32, 32, 4, 8, 8, 2   # ELL is a whole number of 32-value blocks, as a real pack is
    meta = {'model_name': 'fixture-latent', 'family': 'fixture', 'layers': L, 'experts': E,
            'latent_dim': ELL, 'latent_groups': G, 'embedding_length': H, 'build': 'F32',
            'pack_format': 'strata pack index v3'}
    (outdir / 'model.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    labels = rng.integers(0, G, size=(L, E)).astype(np.int32)
    (outdir / 'labels.bin').write_bytes(labels.tobytes())
    # a real Q8_0 file, so the same reader and the same size arithmetic apply as for a released pack
    scale = 1.0 / 64.0
    def write_q8_rows(fh, rows, n_values):
        codes = rng.integers(-127, 128, size=(rows, n_values)).astype(np.int8)
        for r in range(rows):
            for blk in range(n_values // QK):
                fh.write(struct.pack('<e', scale))
                fh.write(codes[r, blk * QK:(blk + 1) * QK].tobytes())
    with (outdir / 'latent_dense_q8.bin').open('wb') as f:
        for _ in range(L):
            write_q8_rows(f, G * ELL, H)
            write_q8_rows(f, G * H, ELL)
    with (outdir / 'experts.bin').open('wb') as f:
        for l in range(L):
            for e in range(E):
                f.write(rng.standard_normal(ELL * FF).astype(np.float32).tobytes())   # gate (as f32)
                f.write(rng.standard_normal(ELL * FF).astype(np.float32).tobytes())   # up
                f.write(rng.standard_normal(FF * ELL).astype(np.float32).tobytes())   # down
    lines = ['# strata native experts v6: fixture, f32 (expert_embd %d, expert_ff %d)' % (ELL, FF)]
    blob = (ELL * FF * 2 + FF * ELL) * 4
    for l in range(L):
        lines.append('%d 0 0 %d %d' % (l, l * E * blob, blob))
    (outdir / 'native_experts.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('fixture written to', outdir, '(layers %d, ELL %d, n_ff %d, experts %d, f32)' % (L, ELL, FF, E))
    return 0


def forward(pack, companion, layer, seed, slots, dump_dir=None, inputs_dir=None) -> int:
    """One latent layer, computed from the pack's bytes and from the companion GGUF, then compared.

    The expert blob is [gate rows | up rows | down rows]; the rows are read separately here because that is how the
    repack splits them, and reading them together is exactly the mistake this function should not repeat.
    """
    root = pathlib.Path(pack)
    meta = json.loads((root / 'model.json').read_text(encoding='utf-8'))
    H = int(meta.get('embedding_length') or 0)
    ELL = int(meta.get('latent_dim') or 0)
    G = int(meta.get('latent_groups') or 8)
    E = int(meta.get('experts') or 512)
    n_layers_meta = int(meta.get('layers') or 0)
    if not ELL:
        raise SystemExit('model.json needs latent_dim')
    if not H:
        # solve it from the D/U file: one layer = groups*ell rows of H + groups*H rows of ell, both (values/32)*row_b
        du_probe = root / 'latent_dense_q8.bin'
        row_b_probe = 34
        if not du_probe.exists():
            du_probe = root / 'latent_dense_q4.bin'
            row_b_probe = 18
        if not (du_probe.exists() and n_layers_meta):
            raise SystemExit('model.json needs embedding_length (or a D/U file to derive it from)')
        per_layer = du_probe.stat().st_size // n_layers_meta
        H = per_layer * 32 // (2 * G * ELL * row_b_probe)
        print('derived embedding_length %d from %s' % (H, du_probe.name))
    manifest = (root / 'native_experts.txt').read_text(encoding='utf-8', errors='replace').splitlines()
    head = manifest[0]
    first = manifest[1].split()
    FF = int(head.split(' expert_ff ', 1)[1].split(')', 1)[0].split()[0]) if ' expert_ff ' in head else 0
    if not FF:
        raise SystemExit('native_experts.txt does not state expert_ff')
    gu_type, dn_type = int(first[1]), int(first[2])
    du_fmt = 8
    du_path = root / 'latent_dense_q8.bin'
    if not du_path.exists():
        du_path = root / 'latent_dense_q4.bin'
        du_fmt = 4

    def rows_of(buf, ggml_type, n_rows, n_values):
        return deq_rows(buf, ggml_type, n_rows, n_values)

    if dump_dir:
        pathlib.Path(dump_dir).mkdir(parents=True, exist_ok=True)   # the per-slot dumps are written inside the loop

    rng = np.random.default_rng(seed)
    if inputs_dir:
        d = pathlib.Path(inputs_dir)
        x = np.fromfile(d / 'x.f32', dtype=np.float32)
        ids = [int(i) for i in np.fromfile(d / 'ids.i32', dtype=np.int32)]
        w = np.fromfile(d / 'weights.f32', dtype=np.float32)
    else:
        x = rng.standard_normal(H).astype(np.float32)
        ids = [int(i) for i in rng.choice(E, size=slots, replace=False)]
        w = np.full(len(ids), 1.0 / len(ids), dtype=np.float32)
    labels = np.frombuffer((root / 'labels.bin').read_bytes(), dtype=np.int32).reshape(-1, E)[layer]

    du = du_path.read_bytes()
    d_row = (H // 32) * (34 if du_fmt == 8 else 18)
    u_row = (ELL // 32) * (34 if du_fmt == 8 else 18)
    per_layer = G * ELL * d_row + G * H * u_row
    base = layer * per_layer
    D = rows_of(du[base:base + G * ELL * d_row], du_fmt, G * ELL, H).reshape(G, ELL, H)
    U = rows_of(du[base + G * ELL * d_row:base + per_layer], du_fmt, G * H, ELL).reshape(G, H, ELL)
    xl = D @ x                                                  # [G, ELL]

    gu_row = tensor_bytes(gu_type, [ELL])                        # bytes of one gate/up row
    dn_row = tensor_bytes(dn_type, [FF])                         # bytes of one down row
    blob = 2 * FF * gu_row + ELL * dn_row
    out_pack = np.zeros(H, dtype=np.float32)
    with open(root / 'experts.bin', 'rb') as f:
        for s_i, e in enumerate(ids):
            g = int(labels[e])
            f.seek(layer * E * blob + e * blob)
            gate_w = rows_of(f.read(FF * gu_row), gu_type, FF, ELL)
            up_w = rows_of(f.read(FF * gu_row), gu_type, FF, ELL)
            down_w = rows_of(f.read(ELL * dn_row), dn_type, ELL, FF)
            a = gate_w @ xl[g]
            b = up_w @ xl[g]
            h = a / (1.0 + np.exp(-a)) * b
            y = down_w @ h
            if dump_dir and s_i == 0:
                pathlib.Path(dump_dir).joinpath('y_slot0.f32').write_bytes(y.astype(np.float32).tobytes())
                pathlib.Path(dump_dir).joinpath('h_slot0.f32').write_bytes(h.astype(np.float32).tobytes())
                pathlib.Path(dump_dir).joinpath('act_slot0.f32').write_bytes(
                    np.ascontiguousarray(xl[g]).astype(np.float32).tobytes())
                pathlib.Path(dump_dir).joinpath('a_slot0.f32').write_bytes(a.astype(np.float32).tobytes())
            out_pack += w[s_i] * (U[g] @ y)

    if dump_dir:
        d = pathlib.Path(dump_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / 'x.f32').write_bytes(x.astype(np.float32).tobytes())
        (d / 'ids.i32').write_bytes(np.asarray(ids, dtype=np.int32).tobytes())
        (d / 'weights.f32').write_bytes(w.astype(np.float32).tobytes())
        (d / 'out.f32').write_bytes(out_pack.astype(np.float32).tobytes())
        (d / 'shapes.txt').write_text('%d %d\n' % (1, len(ids)), encoding='utf-8')
        print('inputs written to', d, '(one token, %d slots, layer %d)' % (len(ids), layer))

    print('layer %d, H %d, ELL %d, G %d, FF %d, experts %s, weight types %d/%d, D/U %s'
          % (layer, H, ELL, G, FF, ids, gu_type, dn_type, du_fmt))
    print('from pack      : sum %.6f  first four %s' % (float(out_pack.sum()), np.round(out_pack[:4], 6)))
    if not companion:
        return 0

    table, data_start = gguf_tensors(companion)
    Dc = rows_of(gguf_tensor_bytes(companion, 'blk.%d.latent_d.weight' % layer, table, data_start),
                 du_fmt, G * ELL, H).reshape(G, ELL, H)
    Uc = rows_of(gguf_tensor_bytes(companion, 'blk.%d.latent_u.weight' % layer, table, data_start),
                 du_fmt, G * H, ELL).reshape(G, H, ELL)
    xlc = Dc @ x
    gate_all = gguf_tensor_bytes(companion, 'blk.%d.ffn_gate_exps.weight' % layer, table, data_start)
    up_all = gguf_tensor_bytes(companion, 'blk.%d.ffn_up_exps.weight' % layer, table, data_start)
    down_all = gguf_tensor_bytes(companion, 'blk.%d.ffn_down_exps.weight' % layer, table, data_start)
    gsz, dsz = FF * gu_row, ELL * dn_row
    out_comp = np.zeros(H, dtype=np.float32)
    for s_i, e in enumerate(ids):
        g = int(labels[e])
        a = rows_of(gate_all[e * gsz:(e + 1) * gsz], gu_type, FF, ELL) @ xlc[g]
        b = rows_of(up_all[e * gsz:(e + 1) * gsz], gu_type, FF, ELL) @ xlc[g]
        h = a / (1.0 + np.exp(-a)) * b
        y = rows_of(down_all[e * dsz:(e + 1) * dsz], dn_type, ELL, FF) @ h
        out_comp += w[s_i] * (Uc[g] @ y)
    print('from companion : sum %.6f  first four %s' % (float(out_comp.sum()), np.round(out_comp[:4], 6)))
    diff = float(np.abs(out_pack - out_comp).max())
    print('max abs difference between the two reads: %.6g' % diff)
    return 0 if diff == 0.0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--fixture', help='directory to write a tiny f32 latent pack into')
    ap.add_argument('--pack', help='an Edge0 pack directory, for --forward')
    ap.add_argument('--companion', help='the repacked companion GGUF, for --forward')
    ap.add_argument('--layer', type=int, default=0)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--slots', type=int, default=2)
    ap.add_argument('--dump-inputs', help='write x/ids/weights/out for another implementation to check against')
    ap.add_argument('--inputs', help='read x/ids/weights from this directory instead of generating them')
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.fixture:
        return fixture(pathlib.Path(args.fixture))
    if args.pack:
        if not args.self_test and not args.fixture:
            return forward(args.pack, args.companion, args.layer, args.seed, args.slots, args.dump_inputs, args.inputs)
    ap.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
